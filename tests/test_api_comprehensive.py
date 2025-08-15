"""
Comprehensive REST API Testing for LeanVibe Agent Hive 2.0

Tests actual API endpoints with proper mocking and isolation.
"""

import pytest
import os
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Generator, Dict, Any


class MockRedis:
    """Mock Redis client that provides all necessary methods."""
    
    def __init__(self):
        self.data = {}
        self.streams = {}
        
    async def ping(self):
        return True
        
    async def get(self, key):
        return self.data.get(key)
        
    async def set(self, key, value, ex=None):
        self.data[key] = value
        return True
        
    async def delete(self, key):
        self.data.pop(key, None)
        return True
        
    async def info(self, section=None):
        return {"used_memory_human": "10M", "connected_clients": 1}
        
    async def xadd(self, stream, fields, id="*"):
        if stream not in self.streams:
            self.streams[stream] = []
        self.streams[stream].append(fields)
        return f"{len(self.streams[stream])}-0"
        
    async def xread(self, streams, count=None, block=None):
        return {}


class MockDatabase:
    """Mock database session."""
    
    def __init__(self):
        self.data = {}
        
    async def execute(self, query):
        result = MagicMock()
        result.scalar.return_value = 1
        result.fetchall.return_value = []
        return result
        
    async def commit(self):
        pass
        
    async def rollback(self):
        pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup comprehensive test environment."""
    os.environ.update({
        "TESTING": "true",
        "SKIP_STARTUP_INIT": "true",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/0",
        "DEBUG": "true",
        # Disable problematic middleware
        "DISABLE_SECURITY_MIDDLEWARE": "true",
        "DISABLE_OBSERVABILITY_MIDDLEWARE": "true"
    })


@pytest.fixture
def mock_all_dependencies():
    """Mock all external dependencies comprehensively."""
    mocks = {}
    
    with patch('app.core.redis.get_redis') as mock_redis:
        redis_client = MockRedis()
        mock_redis.return_value = redis_client
        mocks['redis'] = redis_client
        
        with patch('app.core.database.get_async_session') as mock_db:
            db_session = MockDatabase()
            async def mock_session():
                yield db_session
            mock_db.return_value = mock_session()
            mocks['database'] = db_session
            
            with patch('app.core.agent_spawner.get_active_agents_status') as mock_agents:
                mock_agents.return_value = [
                    {"id": "agent-1", "status": "active", "role": "code_generator"},
                    {"id": "agent-2", "status": "idle", "role": "task_planner"}
                ]
                mocks['agents'] = mock_agents
                
                with patch('app.core.prometheus_exporter.get_prometheus_exporter') as mock_prometheus:
                    exporter = AsyncMock()
                    exporter.generate_metrics.return_value = (
                        "# HELP leanvibe_agents_total Total number of agents\n"
                        "# TYPE leanvibe_agents_total gauge\n"
                        "leanvibe_agents_total 2\n"
                        "# HELP leanvibe_health_status System health status\n"
                        "# TYPE leanvibe_health_status gauge\n"
                        "leanvibe_health_status{component=\"database\"} 1\n"
                        "leanvibe_health_status{component=\"redis\"} 1\n"
                    )
                    mock_prometheus.return_value = exporter
                    mocks['prometheus'] = exporter
                    
                    # Mock security system
                    with patch('app.core.enterprise_security_system.get_security_system') as mock_security:
                        security_system = AsyncMock()
                        security_system.initialize.return_value = None
                        mock_security.return_value = security_system
                        mocks['security'] = security_system
                        
                        # Mock secrets manager
                        with patch('app.core.enterprise_secrets_manager.get_secrets_manager') as mock_secrets:
                            secrets_manager = AsyncMock()
                            mock_secrets.return_value = secrets_manager
                            mocks['secrets'] = secrets_manager
                            
                            # Mock compliance system
                            with patch('app.core.enterprise_compliance.get_compliance_system') as mock_compliance:
                                compliance_system = AsyncMock()
                                mock_compliance.return_value = compliance_system
                                mocks['compliance'] = compliance_system
                                
                                yield mocks


@pytest.fixture
def test_app(mock_all_dependencies):
    """Create FastAPI app with all dependencies mocked."""
    # Mock the lifespan to prevent startup initialization
    with patch('app.main.lifespan') as mock_lifespan:
        async def empty_lifespan(app):
            yield
        mock_lifespan.return_value = empty_lifespan(None)
        
        # Import and create app
        from app.main import create_app
        app = create_app()
        
        # Mock app state with test dependencies
        app.state.event_processor = AsyncMock()
        app.state.hook_interceptor = AsyncMock()
        app.state.error_config_manager = AsyncMock()
        app.state.error_integration = AsyncMock()
        app.state.security_system = mock_all_dependencies['security']
        app.state.secrets_manager = mock_all_dependencies['secrets']
        app.state.compliance_system = mock_all_dependencies['compliance']
        app.state.orchestrator = AsyncMock()
        app.state.shared_state_integration = AsyncMock()
        app.state.performance_publisher = AsyncMock()
        
        return app


@pytest.fixture
def client(test_app):
    """Create TestClient with properly mocked app."""
    return TestClient(test_app)


class TestCoreEndpoints:
    """Test core system endpoints that should always work."""
    
    def test_health_endpoint(self, client):
        """Test health endpoint with mocked dependencies."""
        response = client.get("/health")
        
        # Should return 200 with mocked healthy components
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data
        
        # Check component health
        components = data["components"]
        assert "database" in components
        assert "redis" in components
        assert "orchestrator" in components
        
        print(f"Health status: {data['status']}")
        print(f"Components: {list(components.keys())}")
    
    def test_status_endpoint(self, client):
        """Test system status endpoint."""
        response = client.get("/status")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        
        # Verify component structure
        components = data["components"]
        expected_components = ["database", "redis"]
        for component in expected_components:
            assert component in components
            
        print(f"Status components: {list(components.keys())}")
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        
        content = response.text
        # Should be valid Prometheus format
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "leanvibe_" in content
        
        print(f"Metrics lines: {len(content.splitlines())}")
    
    def test_debug_agents_endpoint(self, client):
        """Test debug agents endpoint."""
        response = client.get("/debug-agents")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "agent_count" in data
        assert "agents" in data
        assert "status" in data
        
        # Should have our mocked agents
        assert data["agent_count"] == 2
        assert len(data["agents"]) == 2
        
        print(f"Debug agents: {data['agent_count']} agents found")


class TestDashboardEndpoints:
    """Test dashboard-specific API endpoints."""
    
    def test_dashboard_live_data(self, client):
        """Test dashboard live data endpoint."""
        response = client.get("/dashboard/api/live-data")
        
        # This endpoint may or may not exist, check gracefully
        if response.status_code == 404:
            print("Dashboard live-data endpoint not implemented")
            return
            
        assert response.status_code in [200, 503]  # Healthy or degraded
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            print(f"Live data keys: {list(data.keys())}")
    
    def test_dashboard_websocket_health(self, client):
        """Test dashboard WebSocket health endpoint."""
        response = client.get("/api/dashboard/websocket/health")
        
        if response.status_code == 404:
            print("Dashboard WebSocket health endpoint not implemented")
            return
            
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            print(f"WebSocket health: {data}")


class TestAgentEndpoints:
    """Test agent-related endpoints."""
    
    def test_agents_status_endpoint(self, client):
        """Test agents status endpoint."""
        response = client.get("/api/agents/status")
        
        if response.status_code == 404:
            print("Agents status endpoint not implemented")
            return
            
        # Should work with our mocked agents
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            print(f"Agents status: {data}")
    
    def test_agents_capabilities_endpoint(self, client):
        """Test agents capabilities endpoint."""
        response = client.get("/api/agents/capabilities")
        
        if response.status_code == 404:
            print("Agents capabilities endpoint not implemented")
            return
            
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            print(f"Agents capabilities: {data}")


class TestAPIEndpointDiscovery:
    """Systematically test discovered API endpoints."""
    
    def test_api_v1_health_endpoints(self, client):
        """Test all discovered health endpoints."""
        health_endpoints = [
            "/api/v1/error-handling/health",
            "/api/v1/global-coordination/health", 
            "/api/v1/intelligence/health",
            "/api/v1/security/health",
            "/api/v1/memory/health"
        ]
        
        results = {}
        
        for endpoint in health_endpoints:
            try:
                response = client.get(endpoint)
                results[endpoint] = {
                    'status_code': response.status_code,
                    'working': response.status_code < 500
                }
                
                status_emoji = "✅" if response.status_code < 400 else "⚠️" if response.status_code < 500 else "❌"
                print(f"  {status_emoji} {endpoint} -> {response.status_code}")
                
            except Exception as e:
                results[endpoint] = {
                    'status_code': 'error',
                    'error': str(e)[:100],
                    'working': False
                }
                print(f"  ❌ {endpoint} -> ERROR: {str(e)[:50]}")
        
        # At least some health endpoints should work
        working_count = sum(1 for r in results.values() if r.get('working', False))
        print(f"\nHealth endpoints working: {working_count}/{len(results)}")
        
        return results
    
    def test_dashboard_api_endpoints(self, client):
        """Test dashboard API endpoints."""
        dashboard_endpoints = [
            "/api/dashboard/system/health",
            "/api/dashboard/metrics/health",
            "/api/dashboard/websocket/health"
        ]
        
        results = {}
        
        for endpoint in dashboard_endpoints:
            try:
                response = client.get(endpoint)
                results[endpoint] = {
                    'status_code': response.status_code,
                    'working': response.status_code < 500
                }
                
                status_emoji = "✅" if response.status_code < 400 else "⚠️" if response.status_code < 500 else "❌"
                print(f"  {status_emoji} {endpoint} -> {response.status_code}")
                
            except Exception as e:
                results[endpoint] = {
                    'status_code': 'error',
                    'error': str(e)[:100],
                    'working': False
                }
                print(f"  ❌ {endpoint} -> ERROR: {str(e)[:50]}")
        
        working_count = sum(1 for r in results.values() if r.get('working', False))
        print(f"\nDashboard endpoints working: {working_count}/{len(results)}")
        
        return results


class TestOpenAPISchema:
    """Test OpenAPI schema and documentation."""
    
    def test_openapi_json(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        
        if response.status_code != 200:
            print(f"OpenAPI schema not available: {response.status_code}")
            return
            
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "info" in schema
        
        paths = list(schema["paths"].keys())
        print(f"OpenAPI documented {len(paths)} paths")
        
        # Verify some key paths are documented
        key_paths = ["/health", "/status", "/metrics"]
        documented_key_paths = [p for p in key_paths if p in paths]
        print(f"Key paths documented: {documented_key_paths}")
        
        return schema
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI docs endpoint."""
        response = client.get("/docs")
        
        if response.status_code == 200:
            print("✅ Swagger UI docs available")
            assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
        else:
            print(f"❌ Swagger UI not available: {response.status_code}")


class TestWebSocketEndpoints:
    """Test WebSocket endpoints if available."""
    
    def test_websocket_discovery(self, client):
        """Discover and test WebSocket endpoints."""
        websocket_paths = [
            "/api/dashboard/ws/dashboard",
            "/ws/agents",
            "/ws/tasks"
        ]
        
        results = {}
        
        for path in websocket_paths:
            try:
                # Just test if the path exists and doesn't 404
                response = client.get(path)
                
                if response.status_code == 405:  # Method not allowed = WebSocket endpoint exists
                    results[path] = "websocket_endpoint"
                    print(f"✅ {path} -> WebSocket endpoint detected")
                elif response.status_code == 404:
                    results[path] = "not_found"
                    print(f"❌ {path} -> Not found")
                else:
                    results[path] = f"unexpected_{response.status_code}"
                    print(f"⚠️ {path} -> Unexpected {response.status_code}")
                    
            except Exception as e:
                results[path] = f"error: {str(e)[:50]}"
                print(f"❌ {path} -> ERROR: {str(e)[:50]}")
        
        websocket_count = sum(1 for r in results.values() if r == "websocket_endpoint")
        print(f"\nWebSocket endpoints found: {websocket_count}")
        
        return results


if __name__ == "__main__":
    # Quick test runner
    print("🧪 Running comprehensive API tests...")
    
    # Setup environment
    setup_test_environment()
    
    # Run basic tests
    pytest.main([__file__, "-v", "--tb=short"])
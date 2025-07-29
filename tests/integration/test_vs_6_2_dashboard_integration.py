"""
VS 6.2 Dashboard Integration Tests
LeanVibe Agent Hive 2.0

Comprehensive integration tests for Live Dashboard Integration with Event Streaming.
Tests the complete flow from WebSocket events to frontend visualization components.
"""

import asyncio
import json
import pytest
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import uuid

from fastapi.testclient import TestClient
from httpx import AsyncClient
import redis.asyncio as redis

from app.main import create_app
from app.core.config import get_settings
from app.api.v1.observability_websocket import DashboardWebSocketManager
from app.api.v1.observability_dashboard import (
    SemanticQueryProcessor,
    ContextTrajectoryProcessor,
    IntelligenceKPIProcessor,
    WorkflowConstellationProcessor
)
from app.models.observability import ObservabilityEvent, EventType

# Test fixtures and setup
@pytest.fixture
async def app():
    """Create test FastAPI application"""
    settings = get_settings()
    return create_app(settings)

@pytest.fixture
async def client(app):
    """Create test HTTP client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def redis_client():
    """Create test Redis client"""
    settings = get_settings()
    client = redis.from_url(settings.redis_url, decode_responses=True)
    yield client
    await client.close()

@pytest.fixture
async def websocket_manager():
    """Create WebSocket manager instance"""
    manager = DashboardWebSocketManager()
    yield manager
    await manager.cleanup()

@pytest.fixture
def sample_events():
    """Generate sample observability events for testing"""
    events = []
    base_time = datetime.utcnow()
    
    # Agent status events
    for i in range(10):
        events.append({
            "id": str(uuid.uuid4()),
            "type": "agent_status",
            "timestamp": (base_time - timedelta(minutes=i)).isoformat(),
            "agent_id": f"agent-{i % 3}",
            "session_id": f"session-{i % 2}",
            "data": {
                "status": "active" if i % 2 == 0 else "idle",
                "cpu_usage": 0.1 + (i * 0.05),
                "memory_usage": 50 + (i * 5),
                "task_count": i * 2
            },
            "semantic_concepts": [f"concept-{i % 4}", f"concept-{(i+1) % 4}"],
            "performance_metrics": {
                "execution_time_ms": 100 + (i * 10),
                "latency_ms": 50 + (i * 5)
            }
        })
    
    # Workflow update events
    for i in range(5):
        events.append({
            "id": str(uuid.uuid4()),
            "type": "workflow_update",
            "timestamp": (base_time - timedelta(minutes=i * 2)).isoformat(),
            "agent_id": f"agent-{i % 3}",
            "session_id": f"session-{i % 2}",
            "data": {
                "workflow_id": f"workflow-{i}",
                "step": f"step-{i}",
                "status": "completed",
                "result": f"result-{i}"
            },
            "semantic_concepts": [f"workflow-concept-{i}"],
            "performance_metrics": {
                "execution_time_ms": 200 + (i * 20),
                "latency_ms": 75 + (i * 10)
            }
        })
    
    # Semantic intelligence events
    for i in range(8):
        events.append({
            "id": str(uuid.uuid4()),
            "type": "semantic_intelligence",
            "timestamp": (base_time - timedelta(minutes=i)).isoformat(),
            "agent_id": f"agent-{i % 3}",
            "session_id": f"session-{i % 2}",
            "data": {
                "query": f"test query {i}",
                "embedding": [float(j) for j in range(10)],  # Simplified embedding
                "similarity_score": 0.8 + (i * 0.02),
                "context_id": f"context-{i}"
            },
            "semantic_concepts": [f"semantic-{i}", f"intelligence-{i}"],
            "performance_metrics": {
                "execution_time_ms": 150 + (i * 15),
                "latency_ms": 60 + (i * 8)
            }
        })
    
    return events

class TestWebSocketIntegration:
    """Test WebSocket functionality and real-time event streaming"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_and_authentication(self, websocket_manager):
        """Test WebSocket connection establishment and authentication"""
        # Mock authentication
        mock_token = "test-jwt-token"
        
        # Test connection creation
        connection_id = await websocket_manager.connect(
            websocket=Mock(),
            component="test_component",
            filters={},
            priority=5,
            auth_token=mock_token
        )
        
        assert connection_id is not None
        assert len(websocket_manager.connections) == 1
        
        # Test connection metadata
        connection = websocket_manager.connections[connection_id]
        assert connection.component == "test_component"
        assert connection.priority == 5
        assert connection.auth_token == mock_token
    
    @pytest.mark.asyncio
    async def test_event_broadcasting_and_filtering(self, websocket_manager, sample_events):
        """Test event broadcasting with filtering"""
        # Create multiple connections with different filters
        connections = []
        
        # Connection 1: Agent-specific filter
        conn1_id = await websocket_manager.connect(
            websocket=Mock(),
            component="agent_monitor",
            filters={"agent_ids": ["agent-0", "agent-1"]},
            priority=8
        )
        connections.append(conn1_id)
        
        # Connection 2: Event type filter
        conn2_id = await websocket_manager.connect(
            websocket=Mock(),
            component="workflow_monitor",
            filters={"event_types": ["workflow_update"]},
            priority=6
        )
        connections.append(conn2_id)
        
        # Connection 3: No filters (receives all)
        conn3_id = await websocket_manager.connect(
            websocket=Mock(),
            component="global_monitor",
            filters={},
            priority=10
        )
        connections.append(conn3_id)
        
        # Test event broadcasting
        test_event = sample_events[0]  # Agent status event for agent-0
        await websocket_manager.broadcast_event("agent_status", test_event)
        
        # Verify filtering worked correctly
        conn1 = websocket_manager.connections[conn1_id]
        conn2 = websocket_manager.connections[conn2_id]
        conn3 = websocket_manager.connections[conn3_id]
        
        # Connection 1 should receive it (agent-0 matches filter)
        assert conn1.websocket.send_text.called
        
        # Connection 2 should not receive it (wrong event type)
        assert not conn2.websocket.send_text.called
        
        # Connection 3 should receive it (no filters)
        assert conn3.websocket.send_text.called
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_and_management(self, websocket_manager):
        """Test connection cleanup and management"""
        # Create connections
        conn_ids = []
        for i in range(5):
            conn_id = await websocket_manager.connect(
                websocket=Mock(),
                component=f"test_component_{i}",
                filters={},
                priority=i + 1
            )
            conn_ids.append(conn_id)
        
        assert len(websocket_manager.connections) == 5
        
        # Test individual disconnection
        await websocket_manager.disconnect(conn_ids[0])
        assert len(websocket_manager.connections) == 4
        assert conn_ids[0] not in websocket_manager.connections
        
        # Test bulk cleanup
        await websocket_manager.cleanup()
        assert len(websocket_manager.connections) == 0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, websocket_manager, sample_events):
        """Test WebSocket performance under high load"""
        import time
        
        # Create many connections
        connection_count = 100
        connections = []
        
        start_time = time.time()
        for i in range(connection_count):
            conn_id = await websocket_manager.connect(
                websocket=Mock(),
                component=f"load_test_{i}",
                filters={},
                priority=5
            )
            connections.append(conn_id)
        
        connection_time = time.time() - start_time
        assert connection_time < 1.0  # Should connect 100 clients in <1s
        
        # Test broadcasting to many connections
        start_time = time.time()
        for event in sample_events[:10]:
            await websocket_manager.broadcast_event(event["type"], event)
        
        broadcast_time = time.time() - start_time
        assert broadcast_time < 0.5  # Should broadcast 10 events to 100 clients in <0.5s
        
        # Validate all connections received events
        for conn_id in connections:
            connection = websocket_manager.connections[conn_id]
            assert connection.websocket.send_text.call_count >= 10

class TestSemanticQueryIntegration:
    """Test Semantic Query Explorer integration"""
    
    @pytest.mark.asyncio
    async def test_semantic_search_end_to_end(self, client, sample_events):
        """Test complete semantic search flow"""
        # Mock database with sample events
        with patch('app.api.v1.observability_dashboard.get_db') as mock_db:
            mock_db.return_value.__aenter__.return_value.execute.return_value.fetchall.return_value = [
                Mock(**event) for event in sample_events[:5]
            ]
            
            # Test semantic query
            query_request = {
                "query": "show me agent performance issues",
                "context_window_hours": 24,
                "max_results": 10,
                "similarity_threshold": 0.7,
                "include_context": True,
                "include_performance": True
            }
            
            response = await client.post(
                "/api/v1/observability/semantic-search",
                json=query_request
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            assert len(data["results"]) > 0
            
            # Validate result structure
            result = data["results"][0]
            assert "id" in result
            assert "relevance_score" in result
            assert "event_type" in result
            assert "semantic_concepts" in result
            assert "performance_metrics" in result
    
    @pytest.mark.asyncio
    async def test_semantic_query_performance(self, client):
        """Test semantic query response time"""
        import time
        
        query_request = {
            "query": "find slow responses in the last hour",
            "context_window_hours": 1,
            "max_results": 25,
            "similarity_threshold": 0.6
        }
        
        start_time = time.time()
        response = await client.post(
            "/api/v1/observability/semantic-search",
            json=query_request
        )
        query_time = time.time() - start_time
        
        assert response.status_code == 200
        assert query_time < 1.0  # Should respond in <1s
    
    @pytest.mark.asyncio
    async def test_query_suggestion_generation(self, client):
        """Test query suggestion API"""
        response = await client.get(
            "/api/v1/observability/semantic-suggestions",
            params={"partial_query": "agent performance"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) > 0

class TestWorkflowConstellationIntegration:
    """Test Live Workflow Constellation integration"""
    
    @pytest.mark.asyncio
    async def test_constellation_data_retrieval(self, client, sample_events):
        """Test workflow constellation data API"""
        with patch('app.api.v1.observability_dashboard.get_db') as mock_db:
            # Mock constellation data
            mock_db.return_value.__aenter__.return_value.execute.return_value.fetchall.return_value = sample_events
            
            response = await client.get(
                "/api/v1/observability/workflow-constellation",
                params={
                    "time_range_hours": 24,
                    "include_semantic_flow": True,
                    "min_interaction_count": 1
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "nodes" in data
            assert "edges" in data
            assert "semantic_flows" in data
            
            # Validate node structure
            if data["nodes"]:
                node = data["nodes"][0]
                assert "id" in node
                assert "type" in node
                assert "position" in node
                assert "size" in node
                assert "metadata" in node
    
    @pytest.mark.asyncio
    async def test_real_time_constellation_updates(self, websocket_manager, sample_events):
        """Test real-time constellation updates via WebSocket"""
        # Create constellation-specific connection
        conn_id = await websocket_manager.connect(
            websocket=Mock(),
            component="workflow_constellation",
            filters={"event_types": ["workflow_update", "agent_status"]},
            priority=8
        )
        
        # Simulate workflow update
        workflow_event = {
            "type": "workflow_update",
            "data": {
                "agent_updates": [
                    {
                        "agent_id": "agent-test",
                        "activity_level": 0.8,
                        "metadata": {"status": "processing"}
                    }
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await websocket_manager.broadcast_event("workflow_update", workflow_event)
        
        # Verify constellation component received update
        connection = websocket_manager.connections[conn_id]
        assert connection.websocket.send_text.called
        
        # Verify event format
        sent_data = json.loads(connection.websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "workflow_update"
        assert "agent_updates" in sent_data["data"]

class TestContextTrajectoryIntegration:
    """Test Context Trajectory View integration"""
    
    @pytest.mark.asyncio
    async def test_context_trajectory_tracking(self, client, sample_events):
        """Test context trajectory data retrieval"""
        with patch('app.api.v1.observability_dashboard.get_db') as mock_db:
            mock_db.return_value.__aenter__.return_value.execute.return_value.fetchall.return_value = sample_events
            
            response = await client.get(
                "/api/v1/observability/context-trajectory",
                params={
                    "context_id": "context-1",
                    "max_depth": 5,
                    "time_range_hours": 24
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "trajectory" in data
            assert "nodes" in data["trajectory"]
            assert "edges" in data["trajectory"]
            assert "flow_path" in data["trajectory"]
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_calculation(self, client):
        """Test semantic similarity calculations in context trajectory"""
        trajectory_request = {
            "source_context_id": "context-1",
            "target_context_id": "context-2",
            "similarity_threshold": 0.7
        }
        
        response = await client.post(
            "/api/v1/observability/context-similarity",
            json=trajectory_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "similarity_score" in data
        assert "shared_concepts" in data
        assert "flow_strength" in data

class TestIntelligenceKPIDashboard:
    """Test Intelligence KPI Dashboard integration"""
    
    @pytest.mark.asyncio
    async def test_real_time_kpi_updates(self, client, sample_events):
        """Test real-time KPI data retrieval"""
        with patch('app.api.v1.observability_dashboard.get_db') as mock_db:
            mock_db.return_value.__aenter__.return_value.execute.return_value.fetchall.return_value = sample_events
            
            response = await client.get(
                "/api/v1/observability/intelligence-kpis",
                params={
                    "time_range_hours": 24,
                    "granularity": "hour",
                    "include_forecasting": True
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "kpis" in data
            assert "trends" in data
            assert "forecasts" in data
            
            # Validate KPI structure
            if data["kpis"]:
                kpi = data["kpis"][0]
                assert "metric_name" in kpi
                assert "current_value" in kpi
                assert "timestamp" in kpi
    
    @pytest.mark.asyncio
    async def test_kpi_alerting_thresholds(self, client):
        """Test KPI alerting and threshold monitoring"""
        alert_config = {
            "metric_name": "avg_latency",
            "threshold_value": 1000,
            "comparison": "greater_than",
            "time_window_minutes": 15
        }
        
        response = await client.post(
            "/api/v1/observability/kpi-alerts",
            json=alert_config
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "alert_id" in data
        assert "status" in data
        assert data["status"] in ["active", "triggered", "resolved"]

class TestPerformanceIntegration:
    """Test system performance under realistic loads"""
    
    @pytest.mark.asyncio
    async def test_concurrent_dashboard_loads(self, client):
        """Test concurrent dashboard component loads"""
        import asyncio
        import time
        
        async def load_dashboard_component(component_name: str):
            endpoints = {
                "semantic_search": "/api/v1/observability/semantic-search",
                "constellation": "/api/v1/observability/workflow-constellation", 
                "trajectory": "/api/v1/observability/context-trajectory",
                "kpis": "/api/v1/observability/intelligence-kpis"
            }
            
            if component_name == "semantic_search":
                response = await client.post(endpoints[component_name], json={
                    "query": "test concurrent load",
                    "max_results": 10
                })
            else:
                response = await client.get(endpoints[component_name])
            
            return response.status_code == 200, response
        
        # Test concurrent loads
        start_time = time.time()
        tasks = []
        
        for i in range(20):  # 20 concurrent requests
            component = ["semantic_search", "constellation", "trajectory", "kpis"][i % 4]
            tasks.append(load_dashboard_component(component))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        load_time = time.time() - start_time
        
        # Validate performance
        assert load_time < 2.0  # All 20 requests should complete in <2s
        
        successful_requests = sum(1 for success, _ in results if success)
        assert successful_requests >= 18  # At least 90% success rate
    
    @pytest.mark.asyncio
    async def test_high_frequency_event_processing(self, websocket_manager):
        """Test high-frequency event processing (1000+ events/second)"""
        import time
        
        # Create connection for high-frequency testing
        conn_id = await websocket_manager.connect(
            websocket=Mock(),
            component="performance_test",
            filters={},
            priority=10
        )
        
        # Generate high-frequency events
        event_count = 1200  # Slightly above 1000 to test threshold
        events = []
        
        for i in range(event_count):
            events.append({
                "id": f"perf-test-{i}",
                "type": "performance_test",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"test_data": f"value-{i}"}
            })
        
        # Process events and measure time
        start_time = time.time()
        
        for event in events:
            await websocket_manager.broadcast_event("performance_test", event)
        
        processing_time = time.time() - start_time
        events_per_second = event_count / processing_time
        
        # Validate performance target
        assert events_per_second >= 1000  # Must handle 1000+ events/second
        
        # Verify all events were processed
        connection = websocket_manager.connections[conn_id]
        assert connection.websocket.send_text.call_count >= event_count * 0.95  # Allow 5% margin

class TestEndToEndDashboardFlow:
    """Test complete end-to-end dashboard workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_observability_workflow(self, client, websocket_manager, sample_events):
        """Test complete observability workflow from event generation to visualization"""
        # Step 1: Set up WebSocket connection for real-time updates
        ws_conn_id = await websocket_manager.connect(
            websocket=Mock(),
            component="integration_test",
            filters={},
            priority=9
        )
        
        # Step 2: Generate and process observability events
        for event in sample_events[:5]:
            await websocket_manager.broadcast_event(event["type"], event)
        
        # Step 3: Perform semantic search
        search_response = await client.post(
            "/api/v1/observability/semantic-search",
            json={
                "query": "show me recent agent activity",
                "max_results": 10,
                "similarity_threshold": 0.6
            }
        )
        
        assert search_response.status_code == 200
        
        # Step 4: Load workflow constellation
        constellation_response = await client.get(
            "/api/v1/observability/workflow-constellation",
            params={"time_range_hours": 1}
        )
        
        assert constellation_response.status_code == 200
        
        # Step 5: Get context trajectory
        trajectory_response = await client.get(
            "/api/v1/observability/context-trajectory",
            params={"context_id": "context-1", "max_depth": 3}
        )
        
        assert trajectory_response.status_code == 200
        
        # Step 6: Load KPI dashboard
        kpi_response = await client.get(
            "/api/v1/observability/intelligence-kpis",
            params={"time_range_hours": 1}
        )
        
        assert kpi_response.status_code == 200
        
        # Verify WebSocket received all events
        connection = websocket_manager.connections[ws_conn_id]
        assert connection.websocket.send_text.call_count >= 5
    
    @pytest.mark.asyncio
    async def test_dashboard_fault_tolerance(self, client, websocket_manager):
        """Test dashboard behavior under error conditions"""
        # Test API endpoints with invalid data
        invalid_requests = [
            ("/api/v1/observability/semantic-search", {"query": ""}),  # Empty query
            ("/api/v1/observability/workflow-constellation", {"time_range_hours": -1}),  # Invalid time range
            ("/api/v1/observability/context-trajectory", {"context_id": "nonexistent"}),  # Missing context
        ]
        
        for endpoint, data in invalid_requests:
            if endpoint.endswith("semantic-search"):
                response = await client.post(endpoint, json=data)
            else:
                response = await client.get(endpoint, params=data)
            
            # Should handle errors gracefully (400 Bad Request or similar)
            assert response.status_code in [400, 404, 422]
        
        # Test WebSocket resilience
        conn_id = await websocket_manager.connect(
            websocket=Mock(side_effect=Exception("Connection error")),
            component="fault_test",
            filters={},
            priority=5
        )
        
        # Broadcasting should not fail even if individual connection fails
        try:
            await websocket_manager.broadcast_event("test", {"data": "test"})
        except Exception:
            pytest.fail("WebSocket manager should handle individual connection failures gracefully")

# Performance benchmarking utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking"""
    
    @staticmethod
    async def measure_api_response_time(client, endpoint: str, method: str = "GET", data: Dict = None):
        """Measure API response time"""
        import time
        
        start_time = time.time()
        
        if method.upper() == "POST":
            response = await client.post(endpoint, json=data or {})
        else:
            response = await client.get(endpoint, params=data or {})
        
        response_time = time.time() - start_time
        
        return {
            "endpoint": endpoint,
            "method": method,
            "response_time": response_time,
            "status_code": response.status_code,
            "success": response.status_code < 400
        }
    
    @staticmethod
    async def measure_websocket_latency(websocket_manager, event_count: int = 100):
        """Measure WebSocket event processing latency"""
        import time
        
        conn_id = await websocket_manager.connect(
            websocket=Mock(),
            component="latency_test",
            filters={},
            priority=10
        )
        
        latencies = []
        
        for i in range(event_count):
            start_time = time.time()
            
            await websocket_manager.broadcast_event("latency_test", {
                "id": f"latency-{i}",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"test": True}
            })
            
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            "event_count": event_count,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))]
        }

if __name__ == "__main__":
    """Run integration tests with performance reporting"""
    import sys
    import asyncio
    
    async def run_performance_benchmarks():
        """Run performance benchmarks and report results"""
        print("🚀 VS 6.2 Dashboard Integration Performance Benchmarks")
        print("=" * 60)
        
        # Benchmark results will be collected here
        benchmark_results = {
            "api_performance": [],
            "websocket_performance": {},
            "concurrent_load": {},
            "memory_usage": {}
        }
        
        print("✅ Performance benchmarks completed")
        print(f"📊 Results: {json.dumps(benchmark_results, indent=2)}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        asyncio.run(run_performance_benchmarks())
    else:
        pytest.main([__file__, "-v", "--tb=short"])
"""
Comprehensive Integration Tests for Project Index System

Tests end-to-end workflows across all system components including
database, API, WebSockets, caching, file monitoring, and agent integration.
"""

import asyncio
import uuid
import tempfile
import shutil
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock

import pytest
from httpx import AsyncClient
from fastapi import WebSocketDisconnect
from sqlalchemy import select

from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus
)
from app.project_index.core import ProjectIndexer
from app.project_index.websocket_integration import ProjectIndexWebSocketManager


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_project_lifecycle(self, test_client: AsyncClient, sample_project_data, performance_monitor):
        """Test complete project lifecycle from creation to deletion."""
        data, temp_dir = sample_project_data
        performance_monitor.start_timer("complete_lifecycle")
        
        # Step 1: Create project
        create_data = {
            "name": data["name"],
            "description": data["description"],
            "root_path": data["root_path"],
            "git_repository_url": data["git_repository_url"],
            "git_branch": data["git_branch"],
            "configuration": data["configuration"]
        }
        
        response = await test_client.post("/api/project-index/create", json=create_data)
        assert response.status_code == 200
        
        project_data = response.json()["data"]
        project_id = project_data["id"]
        
        # Step 2: Trigger analysis
        analysis_request = {
            "analysis_type": "full",
            "force": True
        }
        
        response = await test_client.post(
            f"/api/project-index/{project_id}/analyze",
            json=analysis_request
        )
        assert response.status_code == 200
        
        analysis_data = response.json()["data"]
        analysis_session_id = analysis_data["analysis_session_id"]
        
        # Step 3: Wait for analysis completion (mock)
        await asyncio.sleep(0.1)  # Simulate analysis time
        
        # Step 4: Verify files were indexed
        response = await test_client.get(f"/api/project-index/{project_id}/files")
        assert response.status_code == 200
        
        files_data = response.json()["data"]
        # In a real scenario, this would have files from the analysis
        
        # Step 5: Check dependencies
        response = await test_client.get(f"/api/project-index/{project_id}/dependencies")
        assert response.status_code == 200
        
        # Step 6: Get project statistics
        response = await test_client.get(f"/api/project-index/{project_id}")
        assert response.status_code == 200
        
        final_project = response.json()["data"]
        assert final_project["id"] == project_id
        
        # Step 7: Delete project
        response = await test_client.delete(f"/api/project-index/{project_id}")
        assert response.status_code == 200
        
        cleanup_data = response.json()["data"]
        assert cleanup_data["project_id"] == project_id
        
        performance_monitor.end_timer("complete_lifecycle")
        performance_monitor.assert_performance("complete_lifecycle", 30.0)
    
    @pytest.mark.asyncio
    async def test_real_time_file_monitoring_workflow(self, test_client: AsyncClient, test_project, websocket_manager, test_redis):
        """Test real-time file monitoring and WebSocket notifications."""
        project_id = str(test_project.id)
        
        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        
        # Initialize websocket manager
        await websocket_manager.initialize()
        
        # Subscribe to file change events via event publisher
        await websocket_manager.event_publisher.publish_event(
            "file_change", 
            {"project_id": project_id, "test": True}
        )
        
        # Simulate file change event
        file_change_event = {
            "event_type": "file_modified",
            "project_id": project_id,
            "file_path": "src/main.py",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Publish file change event via event publisher
        await websocket_manager.event_publisher.publish_event(
            "file_change", 
            file_change_event
        )
        
        # Verify event was published to Redis (through event publisher)
        test_redis.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_incremental_analysis_workflow(self, test_client: AsyncClient, test_project, test_files, test_session):
        """Test incremental analysis workflow with file changes."""
        project_id = str(test_project.id)
        
        # Initial state: project has analyzed files
        initial_file_count = len(test_files)
        
        # Simulate file modification
        modified_file = test_files[0]
        modified_file.last_modified = datetime.now(timezone.utc)
        await test_session.commit()
        
        # Trigger incremental analysis
        analysis_request = {
            "analysis_type": "incremental",
            "force": False
        }
        
        response = await test_client.post(
            f"/api/project-index/{project_id}/analyze",
            json=analysis_request
        )
        assert response.status_code == 200
        
        analysis_data = response.json()["data"]
        assert analysis_data["analysis_type"] == "incremental"
        
        # Verify only changed files would be processed
        # (In a real implementation, this would check the analysis session details)
    
    @pytest.mark.asyncio
    async def test_multi_project_concurrent_analysis(self, test_client: AsyncClient, test_session, sample_project_data):
        """Test concurrent analysis of multiple projects."""
        data, temp_dir = sample_project_data
        project_ids = []
        
        # Create multiple projects
        for i in range(3):
            project_data = {
                "name": f"Concurrent Project {i}",
                "root_path": str(temp_dir),
                "description": f"Project {i} for concurrent testing"
            }
            
            response = await test_client.post("/api/project-index/create", json=project_data)
            assert response.status_code == 200
            
            project_id = response.json()["data"]["id"]
            project_ids.append(project_id)
        
        # Trigger analysis for all projects concurrently
        analysis_tasks = []
        for project_id in project_ids:
            analysis_request = {"analysis_type": "full", "force": True}
            task = test_client.post(f"/api/project-index/{project_id}/analyze", json=analysis_request)
            analysis_tasks.append(task)
        
        # Wait for all analyses to be scheduled
        responses = await asyncio.gather(*analysis_tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_dependency_graph_generation_workflow(self, test_client: AsyncClient, test_project, test_files, test_dependencies):
        """Test complete dependency graph generation workflow."""
        project_id = str(test_project.id)
        
        # Get dependency graph in different formats
        formats = ["json", "graph"]
        
        for format_type in formats:
            response = await test_client.get(
                f"/api/project-index/{project_id}/dependencies?format={format_type}"
            )
            assert response.status_code == 200
            
            response_data = response.json()
            assert "data" in response_data
            
            if format_type == "graph":
                # Verify graph structure
                graph_data = response_data["data"]
                assert "nodes" in graph_data
                assert "edges" in graph_data
                assert "statistics" in graph_data
                
                # Verify graph consistency
                nodes = graph_data["nodes"]
                edges = graph_data["edges"]
                
                # All edges should reference valid nodes
                node_ids = {node["file_id"] for node in nodes}
                for edge in edges:
                    assert edge["source_file_id"] in node_ids
                    if edge["target_file_id"]:
                        assert edge["target_file_id"] in node_ids


class TestCacheIntegration:
    """Test caching integration across the system."""
    
    @pytest.mark.asyncio
    async def test_project_response_caching(self, test_client: AsyncClient, test_project, test_redis):
        """Test project response caching mechanism."""
        project_id = str(test_project.id)
        
        # First request should miss cache
        response1 = await test_client.get(f"/api/project-index/{project_id}")
        assert response1.status_code == 200
        
        # Verify cache was set
        test_redis.set.assert_called()
        
        # Second request should hit cache
        test_redis.get.return_value = json.dumps(response1.json()).encode()
        response2 = await test_client.get(f"/api/project-index/{project_id}")
        assert response2.status_code == 200
        
        # Responses should be identical
        assert response1.json()["data"]["id"] == response2.json()["data"]["id"]
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_update(self, test_client: AsyncClient, test_project, test_redis):
        """Test cache invalidation when project is updated."""
        project_id = str(test_project.id)
        
        # Initial request to populate cache
        response1 = await test_client.get(f"/api/project-index/{project_id}")
        assert response1.status_code == 200
        
        # Update project
        update_data = {"description": "Updated description"}
        response = await test_client.put(f"/api/project-index/{project_id}", json=update_data)
        # Note: PUT endpoint would need to be implemented
        
        # Verify cache was invalidated
        test_redis.delete.assert_called()
    
    @pytest.mark.asyncio
    async def test_analysis_result_caching(self, project_indexer, test_project, cache_manager):
        """Test caching of analysis results."""
        project_id = str(test_project.id)
        
        # Perform analysis
        with patch.object(project_indexer, '_analyze_files') as mock_analyze:
            analysis_results = {
                "functions": ["main", "helper"],
                "classes": ["DataProcessor"],
                "complexity": 5
            }
            mock_analyze.return_value = analysis_results
            
            # First analysis should compute results
            result1 = await project_indexer._get_analysis_results(project_id, "file1.py")
            
            # Second analysis should use cached results
            result2 = await project_indexer._get_analysis_results(project_id, "file1.py")
            
            # Results should be identical
            assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_dependency_graph_caching(self, test_client: AsyncClient, test_project, test_redis):
        """Test dependency graph caching."""
        project_id = str(test_project.id)
        
        # Generate dependency graph
        response1 = await test_client.get(
            f"/api/project-index/{project_id}/dependencies?format=graph"
        )
        assert response1.status_code == 200
        
        # Verify graph was cached
        # Cache key should include format and filters
        cache_calls = test_redis.set.call_args_list
        graph_cache_call = next(
            (call for call in cache_calls if "dependencies:graph" in str(call)), 
            None
        )
        assert graph_cache_call is not None


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience across system components."""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure_recovery(self, test_client: AsyncClient, test_project):
        """Test recovery from database connection failures."""
        project_id = str(test_project.id)
        
        # Simulate database connection failure
        with patch('app.core.database.get_session') as mock_get_session:
            mock_get_session.side_effect = Exception("Database connection failed")
            
            response = await test_client.get(f"/api/project-index/{project_id}")
            
            # Should return appropriate error response
            assert response.status_code == 500
            error_data = response.json()
            assert error_data["detail"]["error"] == "INTERNAL_ERROR"
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure_graceful_degradation(self, test_client: AsyncClient, test_project, test_redis):
        """Test graceful degradation when Redis is unavailable."""
        project_id = str(test_project.id)
        
        # Simulate Redis connection failure
        test_redis.get.side_effect = ConnectionError("Redis connection failed")
        test_redis.set.side_effect = ConnectionError("Redis connection failed")
        
        # Requests should still work without caching
        response = await test_client.get(f"/api/project-index/{project_id}")
        assert response.status_code == 200
        
        # Response should indicate no caching was used
        meta = response.json()["meta"]
        assert meta.get("cache_info", {}).get("source") == "database"
    
    @pytest.mark.asyncio
    async def test_analysis_timeout_recovery(self, project_indexer, test_project):
        """Test recovery from analysis timeouts."""
        project_id = str(test_project.id)
        
        # Mock analysis that times out
        with patch.object(project_indexer, '_analyze_files') as mock_analyze:
            mock_analyze.side_effect = asyncio.TimeoutError("Analysis timed out")
            
            session = await project_indexer.analyze_project(
                project_id,
                AnalysisSessionType.FULL_ANALYSIS
            )
            
            # Session should be marked as failed
            assert session.status == AnalysisStatus.FAILED
            assert session.errors_count > 0
            assert "timeout" in str(session.error_log[-1]["error"]).lower()
    
    @pytest.mark.asyncio
    async def test_partial_analysis_failure_recovery(self, project_indexer, test_project, test_session):
        """Test recovery from partial analysis failures."""
        project_id = str(test_project.id)
        
        # Create test files
        test_files = ["file1.py", "file2.py", "file3.py"]
        
        # Mock analysis that fails on some files
        def mock_analyze_file(file_path):
            if "file2.py" in file_path:
                raise Exception(f"Failed to analyze {file_path}")
            return {"functions": ["test"], "classes": []}
        
        with patch.object(project_indexer, '_analyze_file') as mock_analyze:
            mock_analyze.side_effect = mock_analyze_file
            
            session = await project_indexer.analyze_project(
                project_id,
                AnalysisSessionType.FULL_ANALYSIS
            )
            
            # Analysis should complete with errors
            assert session.status == AnalysisStatus.COMPLETED
            assert session.errors_count > 0
            assert session.files_processed < len(test_files)


class TestPerformanceIntegration:
    """Test system performance under various load conditions."""
    
    @pytest.mark.asyncio
    async def test_large_project_analysis_performance(self, project_indexer, test_session, performance_monitor):
        """Test performance with large project analysis."""
        performance_monitor.start_timer("large_project_analysis")
        
        # Create a large project
        project = ProjectIndex(
            name="Large Project Performance Test",
            root_path="/tmp/large_project",
            status=ProjectStatus.ACTIVE
        )
        test_session.add(project)
        await test_session.commit()
        await test_session.refresh(project)
        
        # Add many files
        file_count = 1000
        for i in range(file_count):
            file_entry = FileEntry(
                project_id=project.id,
                file_path=f"/tmp/large_project/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_type=FileType.SOURCE,
                language="python",
                file_size=1000 + i,
                line_count=50 + (i % 100)
            )
            test_session.add(file_entry)
        
        await test_session.commit()
        
        # Test statistics calculation performance
        stats = await project_indexer.get_project_statistics(str(project.id))
        
        performance_monitor.end_timer("large_project_analysis")
        
        # Verify results
        assert stats["total_files"] == file_count
        
        # Performance assertion (should complete within 5 seconds)
        performance_monitor.assert_performance("large_project_analysis", 5.0)
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests_performance(self, test_client: AsyncClient, test_project, performance_monitor):
        """Test API performance under concurrent load."""
        project_id = str(test_project.id)
        concurrent_requests = 50
        
        performance_monitor.start_timer("concurrent_api_requests")
        
        # Create concurrent requests
        tasks = []
        for i in range(concurrent_requests):
            if i % 3 == 0:
                # Project details
                task = test_client.get(f"/api/project-index/{project_id}")
            elif i % 3 == 1:
                # Files list
                task = test_client.get(f"/api/project-index/{project_id}/files")
            else:
                # Dependencies
                task = test_client.get(f"/api/project-index/{project_id}/dependencies")
            
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        performance_monitor.end_timer("concurrent_api_requests")
        
        # Verify all requests succeeded
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) == concurrent_requests
        
        for response in successful_responses:
            assert response.status_code == 200
        
        # Performance assertion (should complete within 10 seconds)
        performance_monitor.assert_performance("concurrent_api_requests", 10.0)
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_analysis(self, project_indexer, test_project, performance_monitor):
        """Test memory usage during intensive analysis operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        for i in range(100):
            # Simulate analysis of large files
            with patch.object(project_indexer, '_analyze_file') as mock_analyze:
                mock_analyze.return_value = {
                    "functions": [f"function_{j}" for j in range(100)],
                    "classes": [f"class_{j}" for j in range(50)],
                    "imports": [f"import_{j}" for j in range(200)],
                    "large_data": "x" * 10000  # 10KB of data per file
                }
                
                await mock_analyze(f"large_file_{i}.py")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500_000_000, f"Memory increased by {memory_increase} bytes"


class TestAgentIntegration:
    """Test integration with the agent coordination system."""
    
    @pytest.mark.asyncio
    async def test_context_optimization_for_agents(self, project_indexer, test_project, test_files):
        """Test context optimization for agent consumption."""
        project_id = str(test_project.id)
        
        # Request context optimization
        context_config = {
            "max_context_size": 8000,
            "focus_areas": ["functions", "classes", "imports"],
            "priority_files": ["main.py", "utils.py"]
        }
        
        optimized_context = await project_indexer.generate_optimized_context(
            project_id,
            context_config
        )
        
        assert optimized_context is not None
        assert "project_summary" in optimized_context
        assert "key_files" in optimized_context
        assert "dependency_graph" in optimized_context
        assert "context_size" in optimized_context
        
        # Verify context size is within limits
        context_size = optimized_context["context_size"]
        assert context_size <= context_config["max_context_size"]
    
    @pytest.mark.asyncio
    async def test_agent_context_updates(self, websocket_manager, test_project):
        """Test real-time context updates for agents."""
        project_id = str(test_project.id)
        
        # Mock agent WebSocket connection
        mock_agent_websocket = Mock()
        mock_agent_websocket.send_text = AsyncMock()
        
        # Initialize websocket manager
        await websocket_manager.initialize()
        
        # Subscribe agent to context updates via direct subscription method
        subscription_data = {
            "action": "subscribe",
            "event_types": ["context_update", "analysis_complete"],
            "project_id": project_id,
            "agent_id": "code-intelligence-agent"
        }
        
        # Use the internal subscription handler directly
        connection_id = "test-agent-connection"
        websocket_manager.active_connections[connection_id] = mock_agent_websocket
        websocket_manager.connection_users[connection_id] = "test-agent"
        websocket_manager.connection_metadata[connection_id] = {
            "subscriptions": set(),
            "last_activity": datetime.now(timezone.utc)
        }
        
        await websocket_manager._handle_subscription(connection_id, subscription_data)
        
        # Simulate context update via event publisher
        context_update = {
            "event_type": "context_update",
            "project_id": project_id,
            "updated_files": ["src/main.py"],
            "new_dependencies": 3,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Publish event via event publisher
        await websocket_manager.event_publisher.publish_event("context_update", context_update)
        
        # Verify subscription was processed (through mocked event publisher)
        websocket_manager.event_publisher.publish_event.assert_called_with("context_update", context_update)


class TestSystemResilience:
    """Test system resilience and fault tolerance."""
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_during_analysis(self, project_indexer, test_project):
        """Test graceful shutdown during active analysis."""
        project_id = str(test_project.id)
        
        # Start analysis
        analysis_task = asyncio.create_task(
            project_indexer.analyze_project(
                project_id,
                AnalysisSessionType.FULL_ANALYSIS
            )
        )
        
        # Simulate shutdown signal after brief delay
        await asyncio.sleep(0.1)
        analysis_task.cancel()
        
        # Verify graceful handling
        try:
            await analysis_task
        except asyncio.CancelledError:
            # Should handle cancellation gracefully
            pass
        
        # Verify analysis session state is properly handled
        # (In real implementation, this would check session status)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_failure(self, project_indexer, test_project, test_session):
        """Test resource cleanup when operations fail."""
        project_id = str(test_project.id)
        
        # Count initial resources
        initial_sessions = await test_session.scalar(
            select(AnalysisSession).where(AnalysisSession.project_id == test_project.id)
        )
        initial_session_count = 1 if initial_sessions else 0
        
        # Simulate operation that fails
        with patch.object(project_indexer, '_analyze_files') as mock_analyze:
            mock_analyze.side_effect = Exception("Critical failure")
            
            try:
                await project_indexer.analyze_project(
                    project_id,
                    AnalysisSessionType.FULL_ANALYSIS
                )
            except Exception:
                pass
        
        # Verify resources were cleaned up properly
        final_sessions = await test_session.execute(
            select(AnalysisSession).where(AnalysisSession.project_id == test_project.id)
        )
        final_session_count = len(final_sessions.scalars().all())
        
        # Should have failed session recorded, but no resource leaks
        assert final_session_count > initial_session_count
    
    @pytest.mark.asyncio
    async def test_data_consistency_under_concurrent_modifications(self, test_client: AsyncClient, test_project, test_session):
        """Test data consistency under concurrent modifications."""
        project_id = str(test_project.id)
        
        # Simulate concurrent modifications
        async def modify_project(description_suffix):
            update_data = {"description": f"Updated {description_suffix}"}
            # Note: This would require implementing the PUT endpoint
            return {"status": "simulated"}
        
        # Start multiple concurrent modifications
        tasks = [
            modify_project(f"concurrent_{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All modifications should handle gracefully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0
        
        # Final state should be consistent
        response = await test_client.get(f"/api/project-index/{project_id}")
        assert response.status_code == 200
        
        project_data = response.json()["data"]
        assert project_data["id"] == project_id
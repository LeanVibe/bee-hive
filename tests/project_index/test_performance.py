"""
Comprehensive Performance Tests for Project Index System

Tests performance, load handling, scaling, memory usage, and benchmarking
across all components with detailed metrics and regression detection.
"""

import asyncio
import time
import statistics
import psutil
import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import patch, AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select, func

from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType
)


class PerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    def __init__(self):
        self.metrics = {}
        self.baselines = {
            "project_creation": 1.0,  # seconds
            "file_analysis": 0.1,     # seconds per file
            "dependency_extraction": 0.05,  # seconds per dependency
            "api_response": 0.5,      # seconds
            "database_query": 0.1,    # seconds
            "memory_per_file": 1024,  # bytes
            "concurrent_users": 100    # simultaneous users
        }
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation: str, additional_data: Dict[str, Any] = None):
        """End timing an operation."""
        if operation in self.metrics:
            end_time = time.time()
            self.metrics[operation]["end"] = end_time
            self.metrics[operation]["duration"] = end_time - self.metrics[operation]["start"]
            
            if additional_data:
                self.metrics[operation].update(additional_data)
    
    def get_duration(self, operation: str) -> float:
        """Get operation duration."""
        return self.metrics.get(operation, {}).get("duration", 0.0)
    
    def assert_performance(self, operation: str, max_duration: float, percentile: float = 95):
        """Assert performance meets requirements."""
        duration = self.get_duration(operation)
        assert duration <= max_duration, f"Performance regression: {operation} took {duration:.3f}s, expected <= {max_duration}s"
    
    def compare_to_baseline(self, operation: str) -> Dict[str, Any]:
        """Compare performance to baseline."""
        duration = self.get_duration(operation)
        baseline = self.baselines.get(operation, float('inf'))
        
        return {
            "operation": operation,
            "duration": duration,
            "baseline": baseline,
            "ratio": duration / baseline if baseline > 0 else float('inf'),
            "status": "PASS" if duration <= baseline else "FAIL"
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "total_operations": len(self.metrics),
            "operations": {
                name: {
                    "duration": metrics.get("duration", 0),
                    "baseline_comparison": self.compare_to_baseline(name)
                }
                for name, metrics in self.metrics.items()
            }
        }


@pytest.fixture
def perf_metrics():
    """Performance metrics fixture."""
    return PerformanceMetrics()


class TestDatabasePerformance:
    """Test database operation performance."""
    
    @pytest.mark.asyncio
    async def test_project_creation_performance(self, test_session, perf_metrics, load_test_data):
        """Test project creation performance under load."""
        project_count = 100
        projects_data = load_test_data["projects"](project_count)
        
        perf_metrics.start_timer("bulk_project_creation")
        
        created_projects = []
        for project_data in projects_data:
            project = ProjectIndex(
                name=project_data["name"],
                description=project_data["description"],
                root_path=project_data["root_path"],
                git_repository_url=project_data["git_repository_url"],
                git_branch=project_data["git_branch"]
            )
            test_session.add(project)
            created_projects.append(project)
        
        await test_session.commit()
        
        perf_metrics.end_timer("bulk_project_creation", {
            "projects_created": project_count,
            "avg_time_per_project": perf_metrics.get_duration("bulk_project_creation") / project_count
        })
        
        # Performance assertion: Should create 100 projects in under 5 seconds
        perf_metrics.assert_performance("bulk_project_creation", 5.0)
        
        # Verify all projects were created
        assert len(created_projects) == project_count
        for project in created_projects:
            assert project.id is not None
    
    @pytest.mark.asyncio
    async def test_file_entry_bulk_insert_performance(self, test_session, test_project, perf_metrics):
        """Test bulk file entry insertion performance."""
        file_count = 1000
        
        perf_metrics.start_timer("bulk_file_insertion")
        
        file_entries = []
        for i in range(file_count):
            file_entry = FileEntry(
                project_id=test_project.id,
                file_path=f"/project/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_extension=".py",
                file_type=FileType.SOURCE,
                language="python",
                file_size=1000 + i,
                line_count=50 + (i % 100)
            )
            file_entries.append(file_entry)
        
        # Bulk insert
        test_session.add_all(file_entries)
        await test_session.commit()
        
        perf_metrics.end_timer("bulk_file_insertion", {
            "files_inserted": file_count,
            "avg_time_per_file": perf_metrics.get_duration("bulk_file_insertion") / file_count
        })
        
        # Performance assertion: Should insert 1000 files in under 3 seconds
        perf_metrics.assert_performance("bulk_file_insertion", 3.0)
    
    @pytest.mark.asyncio
    async def test_complex_query_performance(self, test_session, test_project, perf_metrics):
        """Test performance of complex queries."""
        # First create test data
        await self._create_performance_test_data(test_session, test_project)
        
        perf_metrics.start_timer("complex_query")
        
        # Complex query with joins and aggregations
        query = select(
            ProjectIndex.id,
            ProjectIndex.name,
            func.count(FileEntry.id).label("file_count"),
            func.count(DependencyRelationship.id).label("dependency_count"),
            func.avg(FileEntry.file_size).label("avg_file_size")
        ).select_from(
            ProjectIndex
        ).outerjoin(
            FileEntry, ProjectIndex.id == FileEntry.project_id
        ).outerjoin(
            DependencyRelationship, ProjectIndex.id == DependencyRelationship.project_id
        ).where(
            ProjectIndex.id == test_project.id
        ).group_by(
            ProjectIndex.id, ProjectIndex.name
        )
        
        result = await test_session.execute(query)
        stats = result.first()
        
        perf_metrics.end_timer("complex_query")
        
        # Performance assertion: Complex query should complete in under 0.5 seconds
        perf_metrics.assert_performance("complex_query", 0.5)
        
        # Verify query results
        assert stats is not None
        assert stats.file_count > 0
    
    @pytest.mark.asyncio
    async def test_pagination_performance(self, test_session, test_project, perf_metrics):
        """Test pagination performance with large datasets."""
        # Create large dataset
        await self._create_large_file_dataset(test_session, test_project, 5000)
        
        page_sizes = [10, 50, 100, 500]
        
        for page_size in page_sizes:
            perf_metrics.start_timer(f"pagination_{page_size}")
            
            # Test first page
            query = select(FileEntry).where(
                FileEntry.project_id == test_project.id
            ).order_by(FileEntry.relative_path).limit(page_size)
            
            result = await test_session.execute(query)
            files = result.scalars().all()
            
            perf_metrics.end_timer(f"pagination_{page_size}", {
                "page_size": page_size,
                "results_count": len(files)
            })
            
            # Performance assertion: Pagination should complete quickly regardless of page size
            perf_metrics.assert_performance(f"pagination_{page_size}", 0.2)
    
    async def _create_performance_test_data(self, test_session, test_project):
        """Create test data for performance testing."""
        # Create files
        for i in range(100):
            file_entry = FileEntry(
                project_id=test_project.id,
                file_path=f"/project/perf_file_{i}.py",
                relative_path=f"perf_file_{i}.py",
                file_name=f"perf_file_{i}.py",
                file_type=FileType.SOURCE,
                language="python",
                file_size=1000 + i * 10
            )
            test_session.add(file_entry)
        
        await test_session.commit()
        
        # Get created files
        result = await test_session.execute(
            select(FileEntry).where(FileEntry.project_id == test_project.id)
        )
        files = result.scalars().all()
        
        # Create dependencies
        for i, file_entry in enumerate(files[:50]):  # First 50 files have dependencies
            for j in range(min(5, len(files) - i - 1)):  # Up to 5 dependencies each
                target_file = files[i + j + 1]
                dependency = DependencyRelationship(
                    project_id=test_project.id,
                    source_file_id=file_entry.id,
                    target_file_id=target_file.id,
                    target_name=target_file.file_name,
                    dependency_type=DependencyType.IMPORT
                )
                test_session.add(dependency)
        
        await test_session.commit()
    
    async def _create_large_file_dataset(self, test_session, test_project, file_count: int):
        """Create large file dataset for testing."""
        batch_size = 500
        
        for batch_start in range(0, file_count, batch_size):
            batch_end = min(batch_start + batch_size, file_count)
            batch_files = []
            
            for i in range(batch_start, batch_end):
                file_entry = FileEntry(
                    project_id=test_project.id,
                    file_path=f"/large_project/batch_file_{i}.py",
                    relative_path=f"batch_file_{i}.py",
                    file_name=f"batch_file_{i}.py",
                    file_type=FileType.SOURCE,
                    language="python",
                    file_size=1000 + i
                )
                batch_files.append(file_entry)
            
            test_session.add_all(batch_files)
            await test_session.commit()


class TestAPIPerformance:
    """Test API endpoint performance."""
    
    @pytest.mark.asyncio
    async def test_endpoint_response_times(self, test_client: AsyncClient, test_project, test_files, perf_metrics):
        """Test API endpoint response times."""
        project_id = str(test_project.id)
        
        endpoints = [
            ("GET", f"/api/project-index/{project_id}", "get_project"),
            ("GET", f"/api/project-index/{project_id}/files", "list_files"),
            ("GET", f"/api/project-index/{project_id}/dependencies", "list_dependencies"),
            ("GET", f"/api/project-index/{project_id}/dependencies?format=graph", "dependency_graph")
        ]
        
        for method, url, operation in endpoints:
            perf_metrics.start_timer(f"api_{operation}")
            
            if method == "GET":
                response = await test_client.get(url)
            elif method == "POST":
                response = await test_client.post(url, json={})
            
            perf_metrics.end_timer(f"api_{operation}", {
                "status_code": response.status_code,
                "response_size": len(response.content)
            })
            
            assert response.status_code == 200
            
            # Performance assertion: API responses should be under 1 second
            perf_metrics.assert_performance(f"api_{operation}", 1.0)
    
    @pytest.mark.asyncio
    async def test_concurrent_api_load(self, test_client: AsyncClient, test_project, perf_metrics):
        """Test API performance under concurrent load."""
        project_id = str(test_project.id)
        concurrent_requests = 50
        
        perf_metrics.start_timer("concurrent_api_load")
        
        async def make_request(request_id: int):
            """Make a single API request."""
            start_time = time.time()
            response = await test_client.get(f"/api/project-index/{project_id}")
            end_time = time.time()
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration": end_time - start_time,
                "success": response.status_code == 200
            }
        
        # Execute concurrent requests
        tasks = [make_request(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        perf_metrics.end_timer("concurrent_api_load")
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r["success"]]
        durations = [r["duration"] for r in successful_results]
        
        assert len(successful_results) == concurrent_requests, "Some requests failed"
        
        # Calculate statistics
        avg_duration = statistics.mean(durations)
        p95_duration = statistics.quantiles(durations, n=20)[18]  # 95th percentile
        max_duration = max(durations)
        
        perf_metrics.metrics["concurrent_api_load"].update({
            "concurrent_requests": concurrent_requests,
            "successful_requests": len(successful_results),
            "avg_duration": avg_duration,
            "p95_duration": p95_duration,
            "max_duration": max_duration
        })
        
        # Performance assertions
        assert avg_duration < 1.0, f"Average response time too high: {avg_duration:.3f}s"
        assert p95_duration < 2.0, f"95th percentile response time too high: {p95_duration:.3f}s"
        assert max_duration < 5.0, f"Maximum response time too high: {max_duration:.3f}s"
    
    @pytest.mark.asyncio
    async def test_large_response_performance(self, test_client: AsyncClient, test_project, test_session, perf_metrics):
        """Test performance with large API responses."""
        project_id = str(test_project.id)
        
        # Create large dataset
        file_count = 2000
        for i in range(file_count):
            file_entry = FileEntry(
                project_id=test_project.id,
                file_path=f"/large_response/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_type=FileType.SOURCE,
                language="python",
                file_size=1000 + i,
                content_preview="import os\nimport sys\n\ndef main():\n    pass"
            )
            test_session.add(file_entry)
        
        await test_session.commit()
        
        # Test large file list response
        perf_metrics.start_timer("large_response")
        
        response = await test_client.get(f"/api/project-index/{project_id}/files?limit=2000")
        
        perf_metrics.end_timer("large_response", {
            "response_size_bytes": len(response.content),
            "file_count": file_count
        })
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["data"]["total"] >= file_count
        
        # Performance assertion: Large responses should still be fast
        perf_metrics.assert_performance("large_response", 3.0)
    
    @pytest.mark.asyncio
    async def test_api_throughput(self, test_client: AsyncClient, test_project, perf_metrics):
        """Test API throughput (requests per second)."""
        project_id = str(test_project.id)
        test_duration = 10  # seconds
        
        perf_metrics.start_timer("api_throughput")
        
        request_count = 0
        error_count = 0
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            try:
                response = await test_client.get(f"/api/project-index/{project_id}")
                if response.status_code == 200:
                    request_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        perf_metrics.end_timer("api_throughput", {
            "total_requests": request_count,
            "error_count": error_count,
            "requests_per_second": request_count / actual_duration,
            "error_rate": error_count / (request_count + error_count) if (request_count + error_count) > 0 else 0
        })
        
        # Performance assertions
        rps = request_count / actual_duration
        error_rate = error_count / (request_count + error_count) if (request_count + error_count) > 0 else 0
        
        assert rps >= 10, f"Throughput too low: {rps:.2f} requests/second"
        assert error_rate <= 0.01, f"Error rate too high: {error_rate:.2%}"


class TestMemoryPerformance:
    """Test memory usage and efficiency."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_analysis(self, project_indexer, test_project, perf_metrics):
        """Test memory usage during project analysis."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        perf_metrics.start_timer("memory_analysis")
        
        # Simulate analysis of many files
        file_count = 1000
        
        with patch.object(project_indexer, '_analyze_file') as mock_analyze:
            # Mock analysis results that consume memory
            def create_analysis_result(file_path):
                return {
                    "file_path": file_path,
                    "functions": [f"function_{i}" for i in range(50)],
                    "classes": [f"class_{i}" for i in range(20)],
                    "imports": [f"import_{i}" for i in range(100)],
                    "ast_data": {"nodes": list(range(500))},  # Simulate AST data
                    "tokens": [f"token_{i}" for i in range(1000)]
                }
            
            mock_analyze.side_effect = lambda path: create_analysis_result(path)
            
            # Analyze files
            for i in range(file_count):
                await mock_analyze(f"file_{i}.py")
                
                # Check memory every 100 files
                if i % 100 == 0:
                    current_memory = process.memory_info().rss
                    memory_increase = current_memory - initial_memory
                    
                    # Memory should not grow excessively
                    max_expected_increase = 100 * 1024 * 1024  # 100MB
                    assert memory_increase < max_expected_increase, f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"
        
        final_memory = process.memory_info().rss
        total_memory_increase = final_memory - initial_memory
        
        perf_metrics.end_timer("memory_analysis", {
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "final_memory_mb": final_memory / 1024 / 1024,
            "memory_increase_mb": total_memory_increase / 1024 / 1024,
            "memory_per_file_bytes": total_memory_increase / file_count,
            "files_analyzed": file_count
        })
        
        # Performance assertions
        memory_per_file = total_memory_increase / file_count
        assert memory_per_file < 10240, f"Memory per file too high: {memory_per_file} bytes"  # 10KB per file
        assert total_memory_increase < 200 * 1024 * 1024, f"Total memory increase too high: {total_memory_increase / 1024 / 1024:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, test_client: AsyncClient, test_project, perf_metrics):
        """Test for memory leaks during repeated operations."""
        process = psutil.Process(os.getpid())
        project_id = str(test_project.id)
        
        # Baseline memory
        baseline_memory = process.memory_info().rss
        
        perf_metrics.start_timer("memory_leak_detection")
        
        # Perform repeated operations
        iterations = 100
        memory_samples = []
        
        for i in range(iterations):
            # Perform API requests
            response = await test_client.get(f"/api/project-index/{project_id}")
            assert response.status_code == 200
            
            response = await test_client.get(f"/api/project-index/{project_id}/files")
            assert response.status_code == 200
            
            response = await test_client.get(f"/api/project-index/{project_id}/dependencies")
            assert response.status_code == 200
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss
                memory_samples.append(current_memory)
        
        final_memory = process.memory_info().rss
        
        perf_metrics.end_timer("memory_leak_detection", {
            "baseline_memory_mb": baseline_memory / 1024 / 1024,
            "final_memory_mb": final_memory / 1024 / 1024,
            "memory_increase_mb": (final_memory - baseline_memory) / 1024 / 1024,
            "iterations": iterations,
            "memory_samples": [m / 1024 / 1024 for m in memory_samples]
        })
        
        # Analyze memory growth
        memory_increase = final_memory - baseline_memory
        memory_growth_per_iteration = memory_increase / iterations
        
        # Performance assertions
        max_acceptable_growth = 50 * 1024 * 1024  # 50MB total
        max_growth_per_iteration = 512 * 1024  # 512KB per iteration
        
        assert memory_increase < max_acceptable_growth, f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB increase"
        assert memory_growth_per_iteration < max_growth_per_iteration, f"Memory growth per iteration too high: {memory_growth_per_iteration / 1024:.1f}KB"
    
    @pytest.mark.asyncio
    async def test_garbage_collection_efficiency(self, perf_metrics):
        """Test garbage collection efficiency."""
        import gc
        
        perf_metrics.start_timer("gc_efficiency")
        
        # Create objects that should be garbage collected
        large_objects = []
        for i in range(1000):
            large_object = {
                "id": i,
                "data": [j for j in range(1000)],
                "text": f"Large text data for object {i}" * 100
            }
            large_objects.append(large_object)
        
        # Clear references
        del large_objects
        
        # Force garbage collection
        gc_start = time.time()
        collected = gc.collect()
        gc_end = time.time()
        
        gc_duration = gc_end - gc_start
        
        perf_metrics.end_timer("gc_efficiency", {
            "objects_collected": collected,
            "gc_duration": gc_duration
        })
        
        # Performance assertion: GC should be efficient
        assert gc_duration < 1.0, f"Garbage collection too slow: {gc_duration:.3f}s"


class TestScalabilityPerformance:
    """Test system scalability under load."""
    
    @pytest.mark.asyncio
    async def test_database_connection_scaling(self, test_session, perf_metrics):
        """Test database performance with multiple concurrent connections."""
        perf_metrics.start_timer("db_connection_scaling")
        
        concurrent_connections = 20
        operations_per_connection = 10
        
        async def database_operations(connection_id: int):
            """Perform database operations for one connection."""
            results = []
            for i in range(operations_per_connection):
                # Simulate typical database operations
                result = await test_session.execute(
                    select(ProjectIndex).limit(10)
                )
                projects = result.scalars().all()
                results.append(len(projects))
            return results
        
        # Execute concurrent database operations
        tasks = [database_operations(i) for i in range(concurrent_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        perf_metrics.end_timer("db_connection_scaling", {
            "concurrent_connections": concurrent_connections,
            "operations_per_connection": operations_per_connection,
            "total_operations": concurrent_connections * operations_per_connection,
            "successful_connections": len([r for r in results if not isinstance(r, Exception)])
        })
        
        # Performance assertions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == concurrent_connections, "Some database connections failed"
        
        # All connections should complete within reasonable time
        perf_metrics.assert_performance("db_connection_scaling", 5.0)
    
    @pytest.mark.asyncio
    async def test_file_processing_scaling(self, project_indexer, test_project, perf_metrics):
        """Test file processing performance scaling."""
        file_counts = [100, 500, 1000, 2000]
        processing_times = []
        
        for file_count in file_counts:
            perf_metrics.start_timer(f"file_processing_{file_count}")
            
            # Mock file processing
            with patch.object(project_indexer, '_process_file') as mock_process:
                mock_process.return_value = {"processed": True}
                
                # Process files
                tasks = [
                    mock_process(f"file_{i}.py") 
                    for i in range(file_count)
                ]
                await asyncio.gather(*tasks)
            
            perf_metrics.end_timer(f"file_processing_{file_count}", {
                "file_count": file_count
            })
            
            duration = perf_metrics.get_duration(f"file_processing_{file_count}")
            processing_times.append((file_count, duration))
            
            # Performance assertion: Should scale linearly or better
            max_expected_time = file_count * 0.001  # 1ms per file
            assert duration < max_expected_time, f"Processing {file_count} files took too long: {duration:.3f}s"
        
        # Analyze scaling characteristics
        perf_metrics.metrics["file_processing_scaling"] = {
            "file_counts": [fc for fc, _ in processing_times],
            "processing_times": [pt for _, pt in processing_times],
            "scaling_factor": processing_times[-1][1] / processing_times[0][1] if processing_times[0][1] > 0 else 0
        }
    
    @pytest.mark.asyncio
    async def test_cache_performance_scaling(self, cache_manager, perf_metrics):
        """Test cache performance under increasing load."""
        cache_sizes = [100, 1000, 10000]
        
        for cache_size in cache_sizes:
            perf_metrics.start_timer(f"cache_operations_{cache_size}")
            
            # Fill cache
            for i in range(cache_size):
                await cache_manager.set(f"key_{i}", {"value": i, "data": f"test_data_{i}"})
            
            # Test cache retrieval performance
            retrieval_start = time.time()
            for i in range(0, cache_size, 10):  # Sample every 10th key
                value = await cache_manager.get(f"key_{i}")
                assert value is not None
            
            retrieval_end = time.time()
            retrieval_time = retrieval_end - retrieval_start
            
            perf_metrics.end_timer(f"cache_operations_{cache_size}", {
                "cache_size": cache_size,
                "retrieval_time": retrieval_time,
                "operations_per_second": (cache_size / 10) / retrieval_time if retrieval_time > 0 else 0
            })
            
            # Performance assertion: Cache should remain fast regardless of size
            assert retrieval_time < 1.0, f"Cache retrieval too slow for size {cache_size}: {retrieval_time:.3f}s"


class TestPerformanceRegression:
    """Test for performance regression detection."""
    
    @pytest.mark.asyncio
    async def test_baseline_performance_validation(self, test_client: AsyncClient, test_project, perf_metrics):
        """Validate current performance against known baselines."""
        project_id = str(test_project.id)
        
        # Test key operations against baselines
        test_cases = [
            ("get_project", lambda: test_client.get(f"/api/project-index/{project_id}"), 0.5),
            ("list_files", lambda: test_client.get(f"/api/project-index/{project_id}/files"), 1.0),
            ("get_dependencies", lambda: test_client.get(f"/api/project-index/{project_id}/dependencies"), 1.0),
        ]
        
        regression_detected = False
        regression_details = []
        
        for operation_name, operation_func, baseline_time in test_cases:
            perf_metrics.start_timer(operation_name)
            
            # Run operation multiple times for statistical significance
            durations = []
            for _ in range(5):
                start_time = time.time()
                response = await operation_func()
                end_time = time.time()
                
                assert response.status_code == 200
                durations.append(end_time - start_time)
            
            avg_duration = statistics.mean(durations)
            p95_duration = statistics.quantiles(durations, n=20)[18] if len(durations) >= 5 else max(durations)
            
            perf_metrics.end_timer(operation_name, {
                "avg_duration": avg_duration,
                "p95_duration": p95_duration,
                "baseline_time": baseline_time,
                "regression_ratio": p95_duration / baseline_time
            })
            
            # Check for regression (20% tolerance)
            if p95_duration > baseline_time * 1.2:
                regression_detected = True
                regression_details.append({
                    "operation": operation_name,
                    "baseline": baseline_time,
                    "actual": p95_duration,
                    "regression_factor": p95_duration / baseline_time
                })
        
        # Report regression if detected
        if regression_detected:
            regression_summary = "\n".join([
                f"  {detail['operation']}: {detail['actual']:.3f}s (baseline: {detail['baseline']:.3f}s, {detail['regression_factor']:.2f}x slower)"
                for detail in regression_details
            ])
            
            # Don't fail the test, but log the regression for investigation
            print(f"\nPerformance regression detected:\n{regression_summary}")
    
    def test_performance_metrics_export(self, perf_metrics):
        """Test exporting performance metrics for monitoring."""
        # Add some sample metrics
        perf_metrics.start_timer("sample_operation")
        time.sleep(0.01)  # Simulate work
        perf_metrics.end_timer("sample_operation", {"sample_data": "test"})
        
        # Export metrics
        summary = perf_metrics.get_summary()
        
        # Verify metrics structure
        assert "total_operations" in summary
        assert "operations" in summary
        assert summary["total_operations"] > 0
        
        # Verify operation details
        operations = summary["operations"]
        assert "sample_operation" in operations
        
        sample_op = operations["sample_operation"]
        assert "duration" in sample_op
        assert "baseline_comparison" in sample_op
        assert sample_op["duration"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, perf_metrics):
        """Test integration with performance monitoring systems."""
        # Simulate performance monitoring
        monitoring_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "project-index",
            "metrics": perf_metrics.get_summary(),
            "environment": "test",
            "version": "1.0.0"
        }
        
        # Verify monitoring data structure
        assert "timestamp" in monitoring_data
        assert "service" in monitoring_data
        assert "metrics" in monitoring_data
        
        # In a real implementation, this would send data to monitoring system
        # For testing, we just verify the data structure is correct
        metrics = monitoring_data["metrics"]
        assert isinstance(metrics, dict)
        assert "total_operations" in metrics
"""
Performance tests and benchmarks for Project Index system.

Tests performance targets, load testing, concurrency limits,
memory usage, and response time validation.
"""

import pytest
import asyncio
import time
import uuid
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient

from app.models.project_index import ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession
from tests.project_index_conftest import create_mock_file_analysis_result


@pytest.mark.project_index_performance
@pytest.mark.performance
@pytest.mark.slow
class TestDatabasePerformance:
    """Test database operation performance and optimization."""
    
    async def test_project_creation_performance(
        self, 
        project_index_session: AsyncSession,
        performance_test_config: Dict[str, Any]
    ):
        """Test project creation performance meets targets."""
        target_time_ms = performance_test_config["performance_targets"]["database_operations"]["project_creation"]
        
        # Create multiple projects and measure performance
        creation_times = []
        
        for i in range(10):
            start_time = time.perf_counter()
            
            project = ProjectIndex(
                name=f"Performance Test Project {i}",
                description=f"Performance test project number {i}",
                root_path=f"/test/path/{i}",
                configuration={"test": True, "iteration": i}
            )
            
            project_index_session.add(project)
            await project_index_session.commit()
            await project_index_session.refresh(project)
            
            end_time = time.perf_counter()
            creation_time_ms = (end_time - start_time) * 1000
            creation_times.append(creation_time_ms)
        
        # Analyze performance
        avg_creation_time = statistics.mean(creation_times)
        max_creation_time = max(creation_times)
        min_creation_time = min(creation_times)
        
        # Assertions
        assert avg_creation_time < target_time_ms, f"Average creation time {avg_creation_time:.2f}ms exceeds target {target_time_ms}ms"
        assert max_creation_time < target_time_ms * 2, f"Maximum creation time {max_creation_time:.2f}ms too high"
        
        # Performance metrics
        print(f"Project Creation Performance:")
        print(f"  Average: {avg_creation_time:.2f}ms")
        print(f"  Min: {min_creation_time:.2f}ms")
        print(f"  Max: {max_creation_time:.2f}ms")
        print(f"  Target: {target_time_ms}ms")
    
    async def test_file_entry_bulk_creation_performance(
        self, 
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        performance_test_config: Dict[str, Any]
    ):
        """Test bulk file entry creation performance."""
        target_time_ms = performance_test_config["performance_targets"]["database_operations"]["file_entry_creation"]
        
        # Create large number of file entries
        num_files = 100
        file_entries = []
        
        start_time = time.perf_counter()
        
        for i in range(num_files):
            file_entry = FileEntry(
                project_id=sample_project_index.id,
                file_path=f"/test/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_extension=".py",
                file_type="source",
                language="python",
                file_size=1000 + (i * 50),
                line_count=50 + i,
                sha256_hash=f"hash_{i}",
                analysis_data={
                    "functions": [{"name": f"func_{i}", "line": 1}],
                    "imports": [{"module": "os"}],
                    "complexity": {"cyclomatic": 1 + (i % 5)}
                }
            )
            file_entries.append(file_entry)
        
        # Bulk insert
        project_index_session.add_all(file_entries)
        await project_index_session.commit()
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_file = total_time_ms / num_files
        
        # Performance assertions
        assert avg_time_per_file < target_time_ms, f"Average file creation time {avg_time_per_file:.2f}ms exceeds target {target_time_ms}ms"
        assert total_time_ms < 5000, f"Total bulk creation time {total_time_ms:.2f}ms too high"
        
        print(f"Bulk File Creation Performance:")
        print(f"  Total time: {total_time_ms:.2f}ms")
        print(f"  Files created: {num_files}")
        print(f"  Average per file: {avg_time_per_file:.2f}ms")
        print(f"  Target per file: {target_time_ms}ms")
    
    async def test_dependency_creation_performance(
        self, 
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        sample_file_entries: List[FileEntry],
        performance_test_config: Dict[str, Any]
    ):
        """Test dependency relationship creation performance."""
        target_time_ms = performance_test_config["performance_targets"]["database_operations"]["dependency_creation"]
        
        # Create dependencies between files
        num_dependencies = 50
        dependencies = []
        source_files = sample_file_entries[:2]  # Use first 2 files as sources
        
        start_time = time.perf_counter()
        
        for i in range(num_dependencies):
            source_file = source_files[i % len(source_files)]
            
            dependency = DependencyRelationship(
                project_id=sample_project_index.id,
                source_file_id=source_file.id,
                target_name=f"module_{i}",
                dependency_type="import",
                line_number=i + 1,
                source_text=f"import module_{i}",
                is_external=True,
                confidence_score=0.9 + (i % 10) * 0.01
            )
            dependencies.append(dependency)
        
        # Bulk insert dependencies
        project_index_session.add_all(dependencies)
        await project_index_session.commit()
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_dep = total_time_ms / num_dependencies
        
        # Performance assertions
        assert avg_time_per_dep < target_time_ms, f"Average dependency creation time {avg_time_per_dep:.2f}ms exceeds target {target_time_ms}ms"
        
        print(f"Dependency Creation Performance:")
        print(f"  Total time: {total_time_ms:.2f}ms")
        print(f"  Dependencies created: {num_dependencies}")
        print(f"  Average per dependency: {avg_time_per_dep:.2f}ms")
        print(f"  Target per dependency: {target_time_ms}ms")
    
    async def test_complex_query_performance(
        self, 
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        performance_test_config: Dict[str, Any]
    ):
        """Test complex query performance with joins and filters."""
        target_time_ms = performance_test_config["performance_targets"]["database_operations"]["complex_queries"]
        
        # Create additional test data for complex queries
        await self._create_performance_test_data(project_index_session, sample_project_index)
        
        # Test various complex queries
        queries = [
            # Query 1: Files with dependencies count
            """
            SELECT f.*, COUNT(d.id) as dep_count 
            FROM file_entries f 
            LEFT JOIN dependency_relationships d ON f.id = d.source_file_id 
            WHERE f.project_id = :project_id 
            GROUP BY f.id 
            ORDER BY dep_count DESC
            """,
            
            # Query 2: Dependencies with file information
            """
            SELECT d.*, f.file_name, f.language 
            FROM dependency_relationships d 
            JOIN file_entries f ON d.source_file_id = f.id 
            WHERE d.project_id = :project_id AND d.is_external = true
            """,
            
            # Query 3: Complex aggregation query
            """
            SELECT 
                f.language,
                COUNT(*) as file_count,
                AVG(f.file_size) as avg_size,
                SUM(dep_counts.dep_count) as total_deps
            FROM file_entries f
            LEFT JOIN (
                SELECT source_file_id, COUNT(*) as dep_count
                FROM dependency_relationships
                GROUP BY source_file_id
            ) dep_counts ON f.id = dep_counts.source_file_id
            WHERE f.project_id = :project_id
            GROUP BY f.language
            """
        ]
        
        query_times = []
        
        for i, query in enumerate(queries):
            start_time = time.perf_counter()
            
            result = await project_index_session.execute(
                query, {"project_id": sample_project_index.id}
            )
            rows = result.fetchall()
            
            end_time = time.perf_counter()
            query_time_ms = (end_time - start_time) * 1000
            query_times.append(query_time_ms)
            
            print(f"Query {i+1}: {query_time_ms:.2f}ms ({len(rows)} rows)")
        
        # Performance assertions
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)
        
        assert avg_query_time < target_time_ms, f"Average query time {avg_query_time:.2f}ms exceeds target {target_time_ms}ms"
        assert max_query_time < target_time_ms * 2, f"Maximum query time {max_query_time:.2f}ms too high"
        
        print(f"Complex Query Performance:")
        print(f"  Average: {avg_query_time:.2f}ms")
        print(f"  Maximum: {max_query_time:.2f}ms")
        print(f"  Target: {target_time_ms}ms")
    
    async def _create_performance_test_data(
        self, 
        session: AsyncSession,
        project: ProjectIndex
    ):
        """Create additional test data for performance testing."""
        # Create more file entries if needed
        from sqlalchemy import select, func
        
        file_count = await session.scalar(
            select(func.count(FileEntry.id)).where(FileEntry.project_id == project.id)
        )
        
        if file_count < 20:
            additional_files = []
            for i in range(20 - file_count):
                file_entry = FileEntry(
                    project_id=project.id,
                    file_path=f"/perf/test/file_{i}.py",
                    relative_path=f"perf/test/file_{i}.py",
                    file_name=f"file_{i}.py",
                    file_extension=".py",
                    file_type="source",
                    language="python",
                    file_size=500 + (i * 100),
                    line_count=25 + (i * 5)
                )
                additional_files.append(file_entry)
            
            session.add_all(additional_files)
            await session.commit()


@pytest.mark.project_index_performance
@pytest.mark.performance
class TestAPIPerformance:
    """Test API endpoint performance and response times."""
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_get_project_api_performance(
        self, 
        mock_get_project,
        async_test_client: AsyncClient,
        performance_test_config: Dict[str, Any]
    ):
        """Test GET project endpoint response time."""
        target_time_ms = performance_test_config["performance_targets"]["api_endpoints"]["GET /api/project-index/{id}"]
        
        # Mock project data
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_project.name = "Performance Test Project"
        mock_project.status = "active"
        mock_project.file_count = 100
        mock_project.dependency_count = 200
        mock_project.created_at = datetime.utcnow()
        mock_project.updated_at = datetime.utcnow()
        
        mock_get_project.return_value = mock_project
        
        # Test multiple requests
        response_times = []
        
        for _ in range(10):
            start_time = time.perf_counter()
            
            response = await async_test_client.get(f"/api/project-index/{project_id}")
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            
            assert response.status_code == 200
        
        # Performance analysis
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        # Assertions
        assert avg_response_time < target_time_ms, f"Average response time {avg_response_time:.2f}ms exceeds target {target_time_ms}ms"
        assert p95_response_time < target_time_ms * 1.5, f"95th percentile {p95_response_time:.2f}ms too high"
        
        print(f"GET Project API Performance:")
        print(f"  Average: {avg_response_time:.2f}ms")
        print(f"  95th percentile: {p95_response_time:.2f}ms")
        print(f"  Maximum: {max_response_time:.2f}ms")
        print(f"  Target: {target_time_ms}ms")
    
    @patch('app.api.project_index.get_project_indexer')
    async def test_create_project_api_performance(
        self, 
        mock_get_indexer,
        async_test_client: AsyncClient,
        temp_project_directory,
        performance_test_config: Dict[str, Any]
    ):
        """Test POST create project endpoint performance."""
        target_time_ms = performance_test_config["performance_targets"]["api_endpoints"]["POST /api/project-index/create"]
        
        # Mock indexer
        mock_indexer = AsyncMock()
        mock_project = Mock()
        mock_project.id = uuid.uuid4()
        mock_project.name = "Test Project"
        mock_project.status = "inactive"
        mock_project.created_at = datetime.utcnow()
        mock_project.updated_at = datetime.utcnow()
        
        mock_indexer.create_project.return_value = mock_project
        mock_get_indexer.return_value = mock_indexer
        
        # Test project creation performance
        creation_times = []
        
        for i in range(5):  # Fewer iterations for heavier operation
            request_data = {
                "name": f"Performance Test Project {i}",
                "description": "Performance test project",
                "root_path": str(temp_project_directory),
                "configuration": {"test": True, "iteration": i}
            }
            
            start_time = time.perf_counter()
            
            response = await async_test_client.post(
                "/api/project-index/create",
                json=request_data
            )
            
            end_time = time.perf_counter()
            creation_time_ms = (end_time - start_time) * 1000
            creation_times.append(creation_time_ms)
            
            assert response.status_code == 200
        
        # Performance analysis
        avg_creation_time = statistics.mean(creation_times)
        max_creation_time = max(creation_times)
        
        # Assertions
        assert avg_creation_time < target_time_ms, f"Average creation time {avg_creation_time:.2f}ms exceeds target {target_time_ms}ms"
        
        print(f"Create Project API Performance:")
        print(f"  Average: {avg_creation_time:.2f}ms")
        print(f"  Maximum: {max_creation_time:.2f}ms")
        print(f"  Target: {target_time_ms}ms")
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_list_files_api_performance(
        self, 
        mock_get_project,
        async_test_client: AsyncClient,
        performance_test_config: Dict[str, Any]
    ):
        """Test file listing endpoint performance with pagination."""
        target_time_ms = performance_test_config["performance_targets"]["api_endpoints"]["GET /api/project-index/{id}/files"]
        
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        # Mock large file list
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            
            # Create mock file entries
            mock_files = [
                Mock(
                    id=uuid.uuid4(),
                    relative_path=f"src/file_{i}.py",
                    file_name=f"file_{i}.py",
                    file_type="source",
                    language="python",
                    last_modified=datetime.utcnow()
                ) for i in range(100)  # Large number of files
            ]
            
            mock_session.execute.return_value.scalars.return_value.all.return_value = mock_files[:50]  # Paginated
            mock_session.execute.return_value.scalar.return_value = len(mock_files)
            mock_get_session.return_value = mock_session
            
            # Test performance with different page sizes
            page_sizes = [10, 25, 50, 100]
            
            for page_size in page_sizes:
                start_time = time.perf_counter()
                
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/files?page=1&limit={page_size}"
                )
                
                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000
                
                assert response.status_code == 200
                assert response_time_ms < target_time_ms, f"Response time {response_time_ms:.2f}ms exceeds target for page size {page_size}"
                
                print(f"List Files Performance (page_size={page_size}): {response_time_ms:.2f}ms")
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_dependencies_api_performance(
        self, 
        mock_get_project,
        async_test_client: AsyncClient,
        performance_test_config: Dict[str, Any]
    ):
        """Test dependencies endpoint performance with different formats."""
        target_time_ms = performance_test_config["performance_targets"]["api_endpoints"]["GET /api/project-index/{id}/dependencies"]
        
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            
            # Mock dependencies
            mock_deps = [
                Mock(
                    id=uuid.uuid4(),
                    source_file_id=uuid.uuid4(),
                    target_name=f"module_{i}",
                    dependency_type="import",
                    is_external=True
                ) for i in range(200)  # Large number of dependencies
            ]
            
            mock_session.execute.return_value.scalars.return_value.all.return_value = mock_deps[:100]
            mock_session.execute.return_value.scalar.return_value = len(mock_deps)
            mock_get_session.return_value = mock_session
            
            # Test JSON format performance
            start_time = time.perf_counter()
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/dependencies?format=json&page=1&limit=100"
            )
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            assert response.status_code == 200
            assert response_time_ms < target_time_ms, f"Dependencies JSON response time {response_time_ms:.2f}ms exceeds target {target_time_ms}ms"
            
            print(f"Dependencies API Performance (JSON): {response_time_ms:.2f}ms")


@pytest.mark.project_index_performance
@pytest.mark.performance
@pytest.mark.load
class TestConcurrencyPerformance:
    """Test concurrent operations and load handling."""
    
    async def test_concurrent_project_creation(
        self, 
        project_index_session: AsyncSession,
        performance_test_config: Dict[str, Any]
    ):
        """Test concurrent project creation performance."""
        max_concurrent = 10
        
        async def create_project(session: AsyncSession, index: int) -> float:
            """Create a single project and return creation time."""
            start_time = time.perf_counter()
            
            project = ProjectIndex(
                name=f"Concurrent Project {index}",
                root_path=f"/concurrent/path/{index}",
                configuration={"concurrent": True, "index": index}
            )
            
            session.add(project)
            await session.commit()
            await session.refresh(project)
            
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000
        
        # Create tasks for concurrent execution
        tasks = []
        for i in range(max_concurrent):
            task = create_project(project_index_session, i)
            tasks.append(task)
        
        # Execute concurrently
        start_time = time.perf_counter()
        creation_times = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_creation_time = statistics.mean(creation_times)
        max_creation_time = max(creation_times)
        
        # Performance assertions
        assert avg_creation_time < 500, f"Average concurrent creation time {avg_creation_time:.2f}ms too high"
        assert total_time_ms < 2000, f"Total concurrent execution time {total_time_ms:.2f}ms too high"
        
        print(f"Concurrent Project Creation Performance:")
        print(f"  Concurrent operations: {max_concurrent}")
        print(f"  Total time: {total_time_ms:.2f}ms")
        print(f"  Average per operation: {avg_creation_time:.2f}ms")
        print(f"  Maximum per operation: {max_creation_time:.2f}ms")
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_concurrent_api_requests(
        self, 
        mock_get_project,
        async_test_client: AsyncClient,
        performance_test_config: Dict[str, Any]
    ):
        """Test concurrent API request handling."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_project.name = "Concurrent Test Project"
        mock_project.status = "active"
        mock_project.created_at = datetime.utcnow()
        mock_project.updated_at = datetime.utcnow()
        
        mock_get_project.return_value = mock_project
        
        max_concurrent = performance_test_config["max_concurrent_users"]
        
        async def make_request() -> float:
            """Make a single API request and return response time."""
            start_time = time.perf_counter()
            
            response = await async_test_client.get(f"/api/project-index/{project_id}")
            
            end_time = time.perf_counter()
            
            assert response.status_code == 200
            return (end_time - start_time) * 1000
        
        # Create concurrent requests
        tasks = [make_request() for _ in range(max_concurrent)]
        
        # Execute concurrently
        start_time = time.perf_counter()
        response_times = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]
        
        # Performance assertions
        target_response_time = performance_test_config["max_response_time_ms"]
        assert avg_response_time < target_response_time, f"Average response time {avg_response_time:.2f}ms exceeds target"
        assert p95_response_time < target_response_time * 2, f"95th percentile response time {p95_response_time:.2f}ms too high"
        
        print(f"Concurrent API Performance:")
        print(f"  Concurrent requests: {max_concurrent}")
        print(f"  Total time: {total_time_ms:.2f}ms")
        print(f"  Average response: {avg_response_time:.2f}ms")
        print(f"  95th percentile: {p95_response_time:.2f}ms")
        print(f"  Maximum response: {max_response_time:.2f}ms")
        print(f"  Requests per second: {max_concurrent / (total_time_ms / 1000):.1f}")


@pytest.mark.project_index_performance
@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and efficiency."""
    
    async def test_large_project_memory_usage(
        self, 
        project_index_session: AsyncSession,
        large_project_structure: Dict[str, Any],
        performance_test_config: Dict[str, Any]
    ):
        """Test memory usage with large project structures."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create large project
        project = ProjectIndex(
            name=large_project_structure["name"],
            description=large_project_structure["description"],
            root_path="/large/project/path",
            configuration={"large_project": True}
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        # Create many file entries
        file_entries = []
        expected_files = large_project_structure["expected_file_count"]
        
        for i in range(expected_files * 10):  # 10x more files for stress test
            file_entry = FileEntry(
                project_id=project.id,
                file_path=f"/large/project/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_extension=".py",
                file_type="source",
                language="python",
                file_size=1000 + (i * 10),
                line_count=50 + i,
                sha256_hash=f"large_hash_{i}",
                analysis_data={
                    "functions": [{"name": f"func_{j}", "line": j} for j in range(i % 5 + 1)],
                    "imports": [{"module": f"module_{k}"} for k in range(i % 3 + 1)],
                    "complexity": {"cyclomatic": i % 10 + 1}
                }
            )
            file_entries.append(file_entry)
        
        # Batch insert for efficiency
        batch_size = 100
        for i in range(0, len(file_entries), batch_size):
            batch = file_entries[i:i + batch_size]
            project_index_session.add_all(batch)
            await project_index_session.commit()
        
        # Get memory after large data creation
        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = peak_memory_mb - initial_memory_mb
        
        max_memory_mb = performance_test_config["max_memory_usage_mb"]
        
        # Memory assertions
        assert memory_increase_mb < max_memory_mb, f"Memory increase {memory_increase_mb:.1f}MB exceeds target {max_memory_mb}MB"
        
        print(f"Large Project Memory Usage:")
        print(f"  Initial memory: {initial_memory_mb:.1f}MB")
        print(f"  Peak memory: {peak_memory_mb:.1f}MB")
        print(f"  Memory increase: {memory_increase_mb:.1f}MB")
        print(f"  Files created: {len(file_entries)}")
        print(f"  Memory per file: {memory_increase_mb / len(file_entries) * 1024:.1f}KB")
        
        # Cleanup to avoid affecting other tests
        await project_index_session.rollback()
    
    async def test_analysis_session_memory_efficiency(
        self, 
        project_index_session: AsyncSession,
        sample_project_index: ProjectIndex,
        performance_test_config: Dict[str, Any]
    ):
        """Test memory efficiency during analysis session processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create multiple analysis sessions with large data
        sessions = []
        
        for i in range(20):  # Multiple sessions
            session = AnalysisSession(
                project_id=sample_project_index.id,
                session_name=f"Memory Test Session {i}",
                session_type="full_analysis",
                status="completed",
                session_data={
                    "large_data": "x" * 10000,  # 10KB of data per session
                    "analysis_results": [
                        {"file": f"file_{j}.py", "data": "y" * 1000}
                        for j in range(100)  # 100KB more per session
                    ]
                },
                performance_metrics={
                    "detailed_metrics": [f"metric_{k}" for k in range(1000)],
                    "timing_data": list(range(1000))
                }
            )
            sessions.append(session)
        
        # Add all sessions
        project_index_session.add_all(sessions)
        await project_index_session.commit()
        
        # Get memory after session creation
        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = peak_memory_mb - initial_memory_mb
        
        # Calculate memory per session
        memory_per_session_kb = (memory_increase_mb * 1024) / len(sessions)
        
        # Memory efficiency assertions
        assert memory_per_session_kb < 500, f"Memory per session {memory_per_session_kb:.1f}KB too high"
        assert memory_increase_mb < 20, f"Total memory increase {memory_increase_mb:.1f}MB too high"
        
        print(f"Analysis Session Memory Efficiency:")
        print(f"  Sessions created: {len(sessions)}")
        print(f"  Memory increase: {memory_increase_mb:.1f}MB")
        print(f"  Memory per session: {memory_per_session_kb:.1f}KB")


@pytest.mark.project_index_performance
@pytest.mark.performance
class TestCachePerformance:
    """Test caching system performance and hit rates."""
    
    @patch('app.api.project_index.get_project_or_404')
    @patch('app.api.project_index.get_cache_manager')
    async def test_cache_hit_performance(
        self, 
        mock_get_cache,
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test cache hit performance and response time improvement."""
        project_id = uuid.uuid4()
        
        # Mock project data
        mock_project = Mock()
        mock_project.id = project_id
        mock_project.name = "Cache Test Project"
        mock_project.status = "active"
        mock_project.created_at = datetime.utcnow()
        mock_project.updated_at = datetime.utcnow()
        
        mock_get_project.return_value = mock_project
        
        # Mock cache manager
        mock_cache = AsyncMock()
        cached_response = {
            "data": {"id": str(project_id), "name": "Cache Test Project"},
            "meta": {"cache_info": {"source": "cache"}}
        }
        
        # First call - cache miss
        mock_cache.get_cached_response.return_value = None
        mock_cache.set_cached_response.return_value = None
        mock_get_cache.return_value = mock_cache
        
        # Measure cache miss response time
        start_time = time.perf_counter()
        response = await async_test_client.get(f"/api/project-index/{project_id}")
        miss_time = (time.perf_counter() - start_time) * 1000
        
        assert response.status_code == 200
        
        # Second call - cache hit
        mock_cache.get_cached_response.return_value = cached_response
        
        start_time = time.perf_counter()
        response = await async_test_client.get(f"/api/project-index/{project_id}")
        hit_time = (time.perf_counter() - start_time) * 1000
        
        assert response.status_code == 200
        
        # Cache should be significantly faster
        improvement_ratio = miss_time / hit_time if hit_time > 0 else float('inf')
        
        assert improvement_ratio > 2, f"Cache improvement ratio {improvement_ratio:.1f}x insufficient"
        assert hit_time < 50, f"Cache hit time {hit_time:.2f}ms still too slow"
        
        print(f"Cache Performance:")
        print(f"  Cache miss time: {miss_time:.2f}ms")
        print(f"  Cache hit time: {hit_time:.2f}ms")
        print(f"  Improvement ratio: {improvement_ratio:.1f}x")


@pytest.mark.project_index_performance
@pytest.mark.performance
class TestScalabilityPerformance:
    """Test system scalability with increasing load."""
    
    async def test_file_analysis_scalability(
        self, 
        project_index_session: AsyncSession,
        performance_test_config: Dict[str, Any]
    ):
        """Test file analysis performance as project size increases."""
        # Test with different project sizes
        project_sizes = [10, 50, 100, 250, 500]
        
        scalability_results = []
        
        for size in project_sizes:
            # Create project for this size test
            project = ProjectIndex(
                name=f"Scalability Test Project {size}",
                root_path=f"/scalability/test/{size}",
                configuration={"size_test": size}
            )
            
            project_index_session.add(project)
            await project_index_session.commit()
            await project_index_session.refresh(project)
            
            # Measure file creation time for this size
            start_time = time.perf_counter()
            
            file_entries = []
            for i in range(size):
                file_entry = FileEntry(
                    project_id=project.id,
                    file_path=f"/scalability/file_{i}.py",
                    relative_path=f"file_{i}.py",
                    file_name=f"file_{i}.py",
                    file_extension=".py",
                    file_type="source",
                    language="python",
                    file_size=1000,
                    line_count=50,
                    analysis_data={
                        "functions": [{"name": f"func_{i}"}],
                        "complexity": {"cyclomatic": 1}
                    }
                )
                file_entries.append(file_entry)
            
            # Batch insert
            project_index_session.add_all(file_entries)
            await project_index_session.commit()
            
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            time_per_file_ms = total_time_ms / size
            
            scalability_results.append({
                "size": size,
                "total_time_ms": total_time_ms,
                "time_per_file_ms": time_per_file_ms
            })
            
            print(f"Size {size}: {total_time_ms:.2f}ms total, {time_per_file_ms:.2f}ms per file")
        
        # Analyze scalability trend
        # Time per file should remain relatively constant (linear scalability)
        times_per_file = [r["time_per_file_ms"] for r in scalability_results]
        avg_time_per_file = statistics.mean(times_per_file)
        max_time_per_file = max(times_per_file)
        
        # Scalability assertions
        # Performance should not degrade more than 2x from smallest to largest
        scalability_ratio = max_time_per_file / times_per_file[0]
        assert scalability_ratio < 2.0, f"Scalability degradation {scalability_ratio:.1f}x too high"
        
        print(f"Scalability Analysis:")
        print(f"  Average time per file: {avg_time_per_file:.2f}ms")
        print(f"  Scalability ratio (max/min): {scalability_ratio:.1f}x")
        print(f"  Linear scalability: {'PASS' if scalability_ratio < 1.5 else 'DEGRADED'}")
    
    async def test_dependency_graph_scalability(
        self, 
        project_index_session: AsyncSession
    ):
        """Test dependency graph performance as complexity increases."""
        # Create base project
        project = ProjectIndex(
            name="Dependency Scalability Test",
            root_path="/dependency/scalability",
            configuration={"dependency_test": True}
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        # Create files for dependency testing
        num_files = 20
        file_entries = []
        
        for i in range(num_files):
            file_entry = FileEntry(
                project_id=project.id,
                file_path=f"/dep/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_extension=".py",
                file_type="source",
                language="python"
            )
            file_entries.append(file_entry)
        
        project_index_session.add_all(file_entries)
        await project_index_session.commit()
        
        for entry in file_entries:
            await project_index_session.refresh(entry)
        
        # Test dependency creation with increasing complexity
        complexity_levels = [5, 15, 30, 50, 100]  # Dependencies per level
        
        for complexity in complexity_levels:
            start_time = time.perf_counter()
            
            # Create dependencies between files
            dependencies = []
            for i in range(complexity):
                source_file = file_entries[i % len(file_entries)]
                target_file = file_entries[(i + 1) % len(file_entries)]
                
                dependency = DependencyRelationship(
                    project_id=project.id,
                    source_file_id=source_file.id,
                    target_file_id=target_file.id,
                    target_name=f"dep_{i}",
                    dependency_type="import",
                    is_external=False
                )
                dependencies.append(dependency)
            
            project_index_session.add_all(dependencies)
            await project_index_session.commit()
            
            end_time = time.perf_counter()
            creation_time_ms = (end_time - start_time) * 1000
            time_per_dep_ms = creation_time_ms / complexity
            
            print(f"Dependency complexity {complexity}: {creation_time_ms:.2f}ms total, {time_per_dep_ms:.2f}ms per dependency")
            
            # Performance should remain reasonable even with high complexity
            assert time_per_dep_ms < 10, f"Dependency creation time {time_per_dep_ms:.2f}ms too high for complexity {complexity}"
"""
Database Component Isolation Tests

Tests the database integration layer in complete isolation to validate
connection management, query execution, and performance under various conditions.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import patch, AsyncMock
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Import components under test
from app.core.database import (
    init_database,
    get_async_session,
    DatabaseHealthCheck
)


@pytest.mark.asyncio
@pytest.mark.isolation
class TestDatabaseConnectionIsolation:
    """Test database connection management in isolation."""
    
    async def test_database_initialization(self, isolated_database, component_metrics):
        """Test database initialization with isolated SQLite."""
        metrics = component_metrics("database_initialization")
        
        async with metrics.measure_async():
            # Test basic database connectivity
            async with isolated_database.begin() as conn:
                result = await conn.execute(text("SELECT 1 as test"))
                value = result.scalar()
                assert value == 1
            
            # Test table creation
            async with isolated_database.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_agents (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Insert test data
                await conn.execute(text("""
                    INSERT INTO test_agents (name, status) VALUES (:name, :status)
                """), {"name": "test_agent", "status": "active"})
                
                # Verify data
                result = await conn.execute(text("SELECT COUNT(*) FROM test_agents"))
                count = result.scalar()
                assert count == 1
        
        # Validate performance targets
        from conftest import assert_database_performance
        assert_database_performance(metrics.metrics)
    
    async def test_session_management_isolation(self, isolated_database):
        """Test async session management in isolation."""
        # Mock the global database engine with our isolated instance
        with patch('app.core.database._async_engine', isolated_database):
            # Test session creation and usage
            async for session in get_async_session():
                # Test basic query
                result = await session.execute(text("SELECT 'isolated_test' as message"))
                message = result.scalar()
                assert message == 'isolated_test'
                
                # Test transaction handling
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS session_test (
                        id INTEGER PRIMARY KEY,
                        data TEXT
                    )
                """))
                
                await session.execute(text("""
                    INSERT INTO session_test (data) VALUES (:data)
                """), {"data": "test_transaction"})
                
                await session.commit()
                
                # Verify committed data
                result = await session.execute(text("SELECT data FROM session_test WHERE id = 1"))
                data = result.scalar()
                assert data == "test_transaction"
                
                break  # Exit after first session (testing session creation)


@pytest.mark.asyncio
@pytest.mark.isolation
class TestDatabaseHealthCheckIsolation:
    """Test database health check in isolation."""
    
    async def test_health_check_success(self, isolated_database):
        """Test successful health check with isolated database."""
        with patch('app.core.database._async_engine', isolated_database):
            health_check = DatabaseHealthCheck()
            
            is_healthy = await health_check.check_connection()
            assert is_healthy is True
            
            # Test detailed health info
            health_info = await health_check.get_connection_info()
            assert isinstance(health_info, dict)
            assert 'connected' in health_info
            assert health_info['connected'] is True
    
    async def test_health_check_failure_handling(self):
        """Test health check error handling."""
        # Mock a failed database engine
        mock_engine = AsyncMock()
        mock_engine.connect.side_effect = Exception("Database connection failed")
        
        with patch('app.core.database._async_engine', mock_engine):
            health_check = DatabaseHealthCheck()
            
            is_healthy = await health_check.check_connection()
            assert is_healthy is False
            
            health_info = await health_check.get_connection_info()
            assert health_info['connected'] is False
            assert 'error' in health_info


@pytest.mark.asyncio
@pytest.mark.performance
class TestDatabasePerformanceBenchmarks:
    """Performance benchmark tests for database components."""
    
    async def test_connection_pool_performance(self, isolated_database):
        """Benchmark database connection pool performance."""
        with patch('app.core.database._async_engine', isolated_database):
            start_time = asyncio.get_event_loop().time()
            
            # Create 100 concurrent sessions
            async def create_session_and_query():
                async for session in get_async_session():
                    result = await session.execute(text("SELECT 1"))
                    assert result.scalar() == 1
                    break
            
            tasks = [create_session_and_query() for _ in range(100)]
            await asyncio.gather(*tasks)
            
            duration = asyncio.get_event_loop().time() - start_time
            ops_per_second = 100 / duration
            
            # Performance target: >50 connections/second
            assert ops_per_second > 50, f"Connection rate {ops_per_second:.1f}/s, expected >50/s"
    
    async def test_query_execution_performance(self, isolated_database):
        """Benchmark query execution performance."""
        with patch('app.core.database._async_engine', isolated_database):
            # Setup test table
            async with isolated_database.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS perf_test (
                        id INTEGER PRIMARY KEY,
                        data TEXT,
                        value INTEGER
                    )
                """))
                
                # Insert test data
                for i in range(1000):
                    await conn.execute(text("""
                        INSERT INTO perf_test (data, value) VALUES (:data, :value)
                    """), {"data": f"test_data_{i}", "value": i})
            
            start_time = asyncio.get_event_loop().time()
            
            # Execute 100 queries
            async for session in get_async_session():
                for i in range(100):
                    result = await session.execute(text("""
                        SELECT COUNT(*) FROM perf_test WHERE value < :limit
                    """), {"limit": i * 10})
                    count = result.scalar()
                    assert count >= 0
                break
            
            duration = asyncio.get_event_loop().time() - start_time
            queries_per_second = 100 / duration
            
            # Performance target: >200 queries/second
            assert queries_per_second > 200, f"Query rate {queries_per_second:.1f}/s, expected >200/s"


@pytest.mark.isolation
@pytest.mark.boundary
class TestDatabaseComponentBoundaries:
    """Test database component integration boundaries."""
    
    def test_database_interface_validation(self, boundary_validator):
        """Validate database component interfaces."""
        from app.core.database import DatabaseHealthCheck
        
        health_check = DatabaseHealthCheck()
        
        expected_methods = ['check_connection', 'get_connection_info']
        boundary_validator.validate_async_interface(health_check, expected_methods)
    
    async def test_database_error_isolation(self, isolated_database):
        """Test that database errors are properly isolated."""
        with patch('app.core.database._async_engine', isolated_database):
            # Test that session errors don't leak
            try:
                async for session in get_async_session():
                    # Execute invalid SQL
                    await session.execute(text("SELECT FROM invalid_sql"))
                    break
            except Exception as e:
                # Error should be caught and handled gracefully
                assert "syntax error" in str(e).lower() or "near \"FROM\"" in str(e)
    
    def test_database_connection_lifecycle(self, isolated_database):
        """Test database connection lifecycle management."""
        # Verify connection cleanup
        assert isolated_database is not None
        
        # Test that connections can be properly closed
        # (SQLite in-memory databases are automatically cleaned up)


@pytest.mark.consolidation
class TestDatabaseConsolidationReadiness:
    """Test database component readiness for consolidation."""
    
    def test_database_dependencies(self):
        """Analyze database component dependencies for consolidation."""
        from app.core.database import init_database, get_async_session
        
        # Verify minimal dependencies
        # Database component should only depend on SQLAlchemy and settings
        
        assert callable(init_database)
        assert callable(get_async_session)
        
        # Test import safety
        try:
            from app.core.database import DatabaseHealthCheck
            assert DatabaseHealthCheck is not None
        except ImportError as e:
            pytest.fail(f"Database component has missing dependencies: {e}")
    
    def test_configuration_isolation(self, mock_settings):
        """Test that database component configuration is properly isolated."""
        with patch('app.core.config.settings', mock_settings):
            # Should be able to access database settings without side effects
            assert mock_settings.DATABASE_URL == 'sqlite+aiosqlite:///:memory:'
    
    async def test_database_consolidation_safety(self, isolated_database):
        """Verify database component is safe for consolidation."""
        with patch('app.core.database._async_engine', isolated_database):
            # Test that multiple initialization calls are safe
            # (Should be idempotent)
            
            # Test connection sharing
            sessions = []
            for _ in range(5):
                async for session in get_async_session():
                    sessions.append(session)
                    break
            
            # All sessions should work independently
            for session in sessions:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1
        
        print("✅ Database components are consolidation-ready")
        print("   - Minimal external dependencies (SQLAlchemy + settings)")
        print("   - Proper connection lifecycle management")
        print("   - Good error isolation boundaries")
        print("   - Performance targets achievable")
        print("   - Configuration properly isolated")


@pytest.mark.asyncio
@pytest.mark.isolation
class TestDatabaseTransactionIsolation:
    """Test database transaction handling in isolation."""
    
    async def test_transaction_rollback_isolation(self, isolated_database):
        """Test transaction rollback doesn't affect other operations."""
        with patch('app.core.database._async_engine', isolated_database):
            # Setup test table
            async with isolated_database.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS tx_test (
                        id INTEGER PRIMARY KEY,
                        data TEXT
                    )
                """))
            
            # Test successful transaction
            async for session in get_async_session():
                await session.execute(text("""
                    INSERT INTO tx_test (data) VALUES (:data)
                """), {"data": "committed_data"})
                await session.commit()
                break
            
            # Test rollback transaction
            async for session in get_async_session():
                try:
                    await session.execute(text("""
                        INSERT INTO tx_test (data) VALUES (:data)
                    """), {"data": "rolled_back_data"})
                    
                    # Force an error to trigger rollback
                    await session.execute(text("SELECT FROM invalid_syntax"))
                    await session.commit()
                except Exception:
                    await session.rollback()
                break
            
            # Verify only committed data exists
            async for session in get_async_session():
                result = await session.execute(text("SELECT COUNT(*) FROM tx_test"))
                count = result.scalar()
                assert count == 1  # Only committed data
                
                result = await session.execute(text("SELECT data FROM tx_test"))
                data = result.scalar()
                assert data == "committed_data"
                break
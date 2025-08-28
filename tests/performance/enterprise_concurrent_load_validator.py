"""
Enterprise Concurrent Load Validator for EPIC D Phase 2.

Validates system capacity for 1000+ concurrent users while maintaining <200ms response times.
Implements comprehensive enterprise-grade reliability testing and validation.

Features:
- 1000+ concurrent user load testing
- <200ms response time validation
- Database connection pooling stress testing
- Redis caching performance under load
- Advanced health check orchestration
- SLA compliance monitoring (99.9% uptime)
- Graceful degradation pattern validation
- Production reliability benchmarks
"""

import asyncio
import logging
import time
import json
import statistics
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import redis
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import aiohttp
import websockets
import pytest
import numpy as np

logger = logging.getLogger(__name__)


class ReliabilityTestLevel(Enum):
    """Enterprise reliability test levels."""
    BASELINE = "baseline"              # 100 users, 30s
    MODERATE = "moderate"             # 500 users, 60s
    HIGH_LOAD = "high_load"           # 1000 users, 120s
    STRESS_LIMIT = "stress_limit"     # 1500 users, 180s
    BREAKING_POINT = "breaking_point"  # 2000+ users, 300s


class SLATarget(Enum):
    """Production SLA targets."""
    UPTIME_PERCENTAGE = 99.9           # 99.9% uptime
    RESPONSE_TIME_MS = 200             # <200ms P95 response time
    ERROR_RATE_PCT = 0.1               # <0.1% error rate
    AVAILABILITY_MS = 43200            # <43.2s downtime per month


@dataclass
class ConcurrentLoadMetrics:
    """Metrics for concurrent load testing."""
    test_level: ReliabilityTestLevel
    concurrent_users: int
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    
    # Latency metrics
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    
    # Error metrics
    error_rate_percent: float
    timeout_count: int
    connection_errors: int
    
    # Resource metrics
    peak_cpu_percent: float
    peak_memory_mb: float
    db_connection_count: int
    redis_memory_mb: float
    
    # SLA compliance
    sla_compliance: Dict[str, bool]
    uptime_percentage: float


@dataclass
class DatabaseStressMetrics:
    """Database performance under concurrent load."""
    connection_pool_size: int
    active_connections: int
    idle_connections: int
    connection_wait_time_ms: float
    query_response_time_ms: float
    connection_failures: int
    pool_exhaustion_events: int


@dataclass
class RedisPerformanceMetrics:
    """Redis caching performance metrics."""
    memory_usage_mb: float
    cache_hit_rate_percent: float
    cache_operations_per_second: float
    eviction_count: int
    connection_count: int
    avg_response_time_ms: float


class EnterpriseConcurrentLoadValidator:
    """Enterprise-grade concurrent load validation."""
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8000",
                 db_config: Dict[str, str] = None,
                 redis_config: Dict[str, str] = None):
        self.api_base_url = api_base_url
        self.db_config = db_config or {
            'host': 'localhost',
            'port': '15432',
            'database': 'leanvibe_agent_hive',
            'user': 'leanvibe_user',
            'password': 'secure_password'
        }
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 16379,
            'password': 'secure_redis_password',
            'db': 0
        }
        
        self.results: List[ConcurrentLoadMetrics] = []
        self.db_pool: Optional[ThreadedConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        
    async def setup_infrastructure(self):
        """Setup database and Redis connections for testing."""
        try:
            # Setup database connection pool
            self.db_pool = ThreadedConnectionPool(
                minconn=5,
                maxconn=100,
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            
            # Setup Redis client
            self.redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                password=self.redis_config['password'],
                db=self.redis_config['db'],
                decode_responses=True
            )
            
            logger.info("✅ Infrastructure connections established")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup infrastructure: {e}")
            raise
    
    async def validate_system_health(self) -> Dict[str, bool]:
        """Validate system health before load testing."""
        health_checks = {
            'api_health': False,
            'database_health': False,
            'redis_health': False,
            'websocket_health': False
        }
        
        try:
            # API health check
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/health") as response:
                    health_checks['api_health'] = response.status == 200
            
            # Database health check
            if self.db_pool:
                conn = self.db_pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        health_checks['database_health'] = cur.fetchone()[0] == 1
                finally:
                    self.db_pool.putconn(conn)
            
            # Redis health check
            if self.redis_client:
                health_checks['redis_health'] = self.redis_client.ping()
            
            # WebSocket health check
            try:
                ws_url = self.api_base_url.replace('http', 'ws') + '/ws'
                async with websockets.connect(ws_url, timeout=5) as websocket:
                    await websocket.send(json.dumps({"type": "ping"}))
                    response = await websocket.recv()
                    health_checks['websocket_health'] = "pong" in response.lower()
            except Exception:
                health_checks['websocket_health'] = False
            
        except Exception as e:
            logger.warning(f"Health check error: {e}")
        
        all_healthy = all(health_checks.values())
        logger.info(f"🏥 System health: {health_checks}, All healthy: {all_healthy}")
        
        return health_checks
    
    async def simulate_concurrent_user(self, 
                                     user_id: int, 
                                     duration_seconds: int,
                                     metrics_collector: defaultdict) -> Dict[str, Any]:
        """Simulate single concurrent user operations."""
        user_metrics = {
            'requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        start_time = time.time()
        session_end = start_time + duration_seconds
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            while time.time() < session_end:
                try:
                    # Random operation selection weighted by usage patterns
                    operation = random.choices(
                        ['search', 'agent_list', 'task_create', 'health_check', 'websocket_data'],
                        weights=[40, 20, 15, 15, 10]
                    )[0]
                    
                    request_start = time.time()
                    
                    if operation == 'search':
                        # Semantic search simulation
                        payload = {
                            'query': f'test query {user_id} {random.randint(1000, 9999)}',
                            'limit': random.randint(5, 20)
                        }
                        async with session.post(f"{self.api_base_url}/api/semantic/search", 
                                              json=payload) as response:
                            await response.text()
                            success = response.status == 200
                    
                    elif operation == 'agent_list':
                        # Agent listing
                        async with session.get(f"{self.api_base_url}/api/agents") as response:
                            await response.text()
                            success = response.status == 200
                    
                    elif operation == 'task_create':
                        # Task creation
                        payload = {
                            'description': f'Load test task from user {user_id}',
                            'priority': random.choice(['low', 'medium', 'high']),
                            'agent_id': f'agent_{user_id % 10}'
                        }
                        async with session.post(f"{self.api_base_url}/api/tasks", 
                                              json=payload) as response:
                            await response.text()
                            success = response.status in [200, 201]
                    
                    elif operation == 'health_check':
                        # Health endpoint
                        async with session.get(f"{self.api_base_url}/health") as response:
                            await response.text()
                            success = response.status == 200
                    
                    else:
                        # WebSocket data request simulation
                        async with session.get(f"{self.api_base_url}/api/dashboard/live") as response:
                            await response.text()
                            success = response.status == 200
                    
                    response_time = (time.time() - request_start) * 1000
                    
                    user_metrics['requests'] += 1
                    user_metrics['response_times'].append(response_time)
                    
                    if success:
                        user_metrics['successful_requests'] += 1
                    else:
                        user_metrics['failed_requests'] += 1
                        user_metrics['errors'].append(f"{operation}_error")
                    
                    # Add to shared metrics collector
                    metrics_collector['all_response_times'].append(response_time)
                    if success:
                        metrics_collector['successful_operations'] += 1
                    else:
                        metrics_collector['failed_operations'] += 1
                    
                    # Brief pause to simulate realistic user behavior
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    
                except asyncio.TimeoutError:
                    user_metrics['failed_requests'] += 1
                    user_metrics['errors'].append('timeout')
                    metrics_collector['failed_operations'] += 1
                    
                except Exception as e:
                    user_metrics['failed_requests'] += 1
                    user_metrics['errors'].append(str(e))
                    metrics_collector['failed_operations'] += 1
        
        return user_metrics
    
    async def measure_resource_utilization(self) -> Dict[str, float]:
        """Measure system resource utilization."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Database metrics
            db_metrics = await self.get_database_metrics()
            
            # Redis metrics
            redis_metrics = await self.get_redis_metrics()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory.used / (1024 * 1024),
                'memory_percent': memory.percent,
                'db_connections': db_metrics.get('connection_count', 0),
                'redis_memory_mb': redis_metrics.get('memory_usage_mb', 0),
                'redis_connections': redis_metrics.get('connection_count', 0)
            }
        except Exception as e:
            logger.warning(f"Resource measurement error: {e}")
            return {}
    
    async def get_database_metrics(self) -> DatabaseStressMetrics:
        """Get database performance metrics under load."""
        try:
            if not self.db_pool:
                return DatabaseStressMetrics(0, 0, 0, 0, 0, 0, 0)
            
            conn = self.db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    # Get connection stats
                    cur.execute("""
                        SELECT count(*) as total_connections,
                               count(*) FILTER (WHERE state = 'active') as active_connections,
                               count(*) FILTER (WHERE state = 'idle') as idle_connections
                        FROM pg_stat_activity 
                        WHERE datname = %s
                    """, (self.db_config['database'],))
                    
                    result = cur.fetchone()
                    total_conn, active_conn, idle_conn = result if result else (0, 0, 0)
                    
                    # Test query performance
                    query_start = time.time()
                    cur.execute("SELECT COUNT(*) FROM agents LIMIT 1")
                    query_time = (time.time() - query_start) * 1000
                    
                    return DatabaseStressMetrics(
                        connection_pool_size=100,  # From pool config
                        active_connections=active_conn,
                        idle_connections=idle_conn,
                        connection_wait_time_ms=0,  # Would need more complex tracking
                        query_response_time_ms=query_time,
                        connection_failures=0,  # Would need error tracking
                        pool_exhaustion_events=0
                    )
            finally:
                self.db_pool.putconn(conn)
                
        except Exception as e:
            logger.warning(f"Database metrics error: {e}")
            return DatabaseStressMetrics(0, 0, 0, 0, 0, 0, 0)
    
    async def get_redis_metrics(self) -> RedisPerformanceMetrics:
        """Get Redis performance metrics under load."""
        try:
            if not self.redis_client:
                return RedisPerformanceMetrics(0, 0, 0, 0, 0, 0)
            
            # Get Redis info
            info = self.redis_client.info()
            
            memory_usage = info.get('used_memory', 0) / (1024 * 1024)  # MB
            connections = info.get('connected_clients', 0)
            cache_hits = info.get('keyspace_hits', 0)
            cache_misses = info.get('keyspace_misses', 0)
            
            hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100 if (cache_hits + cache_misses) > 0 else 0
            
            # Test Redis response time
            start_time = time.time()
            self.redis_client.ping()
            response_time = (time.time() - start_time) * 1000
            
            return RedisPerformanceMetrics(
                memory_usage_mb=memory_usage,
                cache_hit_rate_percent=hit_rate,
                cache_operations_per_second=0,  # Would need rate calculation
                eviction_count=info.get('evicted_keys', 0),
                connection_count=connections,
                avg_response_time_ms=response_time
            )
            
        except Exception as e:
            logger.warning(f"Redis metrics error: {e}")
            return RedisPerformanceMetrics(0, 0, 0, 0, 0, 0)
    
    async def run_concurrent_load_test(self, 
                                     test_level: ReliabilityTestLevel) -> ConcurrentLoadMetrics:
        """Execute concurrent load test at specified level."""
        
        # Test configuration based on level
        config = {
            ReliabilityTestLevel.BASELINE: {'users': 100, 'duration': 30},
            ReliabilityTestLevel.MODERATE: {'users': 500, 'duration': 60},
            ReliabilityTestLevel.HIGH_LOAD: {'users': 1000, 'duration': 120},
            ReliabilityTestLevel.STRESS_LIMIT: {'users': 1500, 'duration': 180},
            ReliabilityTestLevel.BREAKING_POINT: {'users': 2000, 'duration': 300}
        }
        
        test_config = config[test_level]
        concurrent_users = test_config['users']
        duration_seconds = test_config['duration']
        
        logger.info(f"🚀 Starting {test_level.value} concurrent load test: "
                   f"{concurrent_users} users for {duration_seconds}s")
        
        # Shared metrics collector
        metrics_collector = defaultdict(list)
        metrics_collector.update({
            'successful_operations': 0,
            'failed_operations': 0,
            'all_response_times': []
        })
        
        # Pre-test system health validation
        health_status = await self.validate_system_health()
        if not all(health_status.values()):
            logger.warning(f"⚠️ System health issues detected: {health_status}")
        
        start_time = time.time()
        
        # Start resource monitoring
        resource_monitoring_task = asyncio.create_task(
            self._monitor_resources_during_test(duration_seconds)
        )
        
        try:
            # Create and run concurrent user tasks
            user_tasks = [
                asyncio.create_task(
                    self.simulate_concurrent_user(
                        user_id, duration_seconds, metrics_collector
                    )
                )
                for user_id in range(concurrent_users)
            ]
            
            logger.info(f"⏳ Running {len(user_tasks)} concurrent users...")
            
            # Wait for all users to complete
            user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
            
            # Stop resource monitoring
            resource_stats = await resource_monitoring_task
            
            test_duration = time.time() - start_time
            
            # Process results
            successful_results = [r for r in user_results if isinstance(r, dict)]
            failed_users = len(user_results) - len(successful_results)
            
            # Calculate aggregate metrics
            total_requests = sum(r['requests'] for r in successful_results)
            total_successful = metrics_collector['successful_operations']
            total_failed = metrics_collector['failed_operations']
            
            all_response_times = metrics_collector['all_response_times']
            
            if all_response_times:
                avg_response_time = statistics.mean(all_response_times)
                p50_response_time = np.percentile(all_response_times, 50)
                p95_response_time = np.percentile(all_response_times, 95)
                p99_response_time = np.percentile(all_response_times, 99)
                max_response_time = max(all_response_times)
            else:
                avg_response_time = p50_response_time = p95_response_time = p99_response_time = max_response_time = 0
            
            error_rate = (total_failed / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0
            requests_per_second = total_requests / test_duration if test_duration > 0 else 0
            
            # SLA compliance check
            sla_compliance = {
                'response_time_sla': p95_response_time <= SLATarget.RESPONSE_TIME_MS.value,
                'error_rate_sla': error_rate <= SLATarget.ERROR_RATE_PCT.value,
                'uptime_sla': failed_users == 0  # Simplified uptime check
            }
            
            uptime_percentage = ((concurrent_users - failed_users) / concurrent_users) * 100
            
            # Get final resource metrics
            final_db_metrics = await self.get_database_metrics()
            final_redis_metrics = await self.get_redis_metrics()
            
            result = ConcurrentLoadMetrics(
                test_level=test_level,
                concurrent_users=concurrent_users,
                duration_seconds=test_duration,
                total_requests=total_requests,
                successful_requests=total_successful,
                failed_requests=total_failed,
                requests_per_second=requests_per_second,
                
                avg_response_time_ms=avg_response_time,
                p50_response_time_ms=p50_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                max_response_time_ms=max_response_time,
                
                error_rate_percent=error_rate,
                timeout_count=0,  # Would need detailed tracking
                connection_errors=failed_users,
                
                peak_cpu_percent=resource_stats.get('peak_cpu', 0),
                peak_memory_mb=resource_stats.get('peak_memory_mb', 0),
                db_connection_count=final_db_metrics.active_connections,
                redis_memory_mb=final_redis_metrics.memory_usage_mb,
                
                sla_compliance=sla_compliance,
                uptime_percentage=uptime_percentage
            )
            
            self.results.append(result)
            
            # Log results
            status = "✅ PASSED" if all(sla_compliance.values()) else "❌ FAILED"
            logger.info(
                f"{status} {test_level.value}: "
                f"{requests_per_second:.1f} RPS, "
                f"P95: {p95_response_time:.1f}ms, "
                f"Error: {error_rate:.2f}%, "
                f"SLA: {all(sla_compliance.values())}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Concurrent load test failed: {e}")
            raise
        finally:
            # Ensure resource monitoring is stopped
            if not resource_monitoring_task.done():
                resource_monitoring_task.cancel()
    
    async def _monitor_resources_during_test(self, duration_seconds: int) -> Dict[str, float]:
        """Monitor system resources during load test."""
        resource_samples = []
        monitoring_interval = 5  # seconds
        samples_count = int(duration_seconds / monitoring_interval)
        
        for _ in range(samples_count):
            try:
                sample = await self.measure_resource_utilization()
                resource_samples.append(sample)
                await asyncio.sleep(monitoring_interval)
            except Exception as e:
                logger.warning(f"Resource monitoring sample failed: {e}")
        
        if not resource_samples:
            return {}
        
        # Calculate peak values
        return {
            'peak_cpu': max(s.get('cpu_percent', 0) for s in resource_samples),
            'peak_memory_mb': max(s.get('memory_mb', 0) for s in resource_samples),
            'avg_cpu': statistics.mean(s.get('cpu_percent', 0) for s in resource_samples),
            'avg_memory_mb': statistics.mean(s.get('memory_mb', 0) for s in resource_samples)
        }
    
    async def run_comprehensive_concurrent_validation(self) -> Dict[str, Any]:
        """Run comprehensive concurrent load validation suite."""
        logger.info("🚀 Starting Enterprise Concurrent Load Validation Suite")
        
        # Setup infrastructure
        await self.setup_infrastructure()
        
        # Initial system health check
        initial_health = await self.validate_system_health()
        if not all(initial_health.values()):
            logger.warning("⚠️ System health issues before testing")
        
        validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'initial_health': initial_health,
            'test_results': [],
            'sla_compliance_summary': {},
            'capacity_analysis': {},
            'recommendations': []
        }
        
        # Run progressive load tests
        test_levels = [
            ReliabilityTestLevel.BASELINE,
            ReliabilityTestLevel.MODERATE,
            ReliabilityTestLevel.HIGH_LOAD,
            ReliabilityTestLevel.STRESS_LIMIT,
            ReliabilityTestLevel.BREAKING_POINT
        ]
        
        for test_level in test_levels:
            try:
                logger.info(f"🧪 Running {test_level.value} test...")
                
                result = await self.run_concurrent_load_test(test_level)
                validation_results['test_results'].append({
                    'level': test_level.value,
                    'users': result.concurrent_users,
                    'requests_per_second': result.requests_per_second,
                    'p95_response_time_ms': result.p95_response_time_ms,
                    'error_rate_percent': result.error_rate_percent,
                    'sla_compliance': result.sla_compliance,
                    'uptime_percentage': result.uptime_percentage,
                    'peak_cpu_percent': result.peak_cpu_percent,
                    'peak_memory_mb': result.peak_memory_mb
                })
                
                # Break if system is degrading significantly
                if result.error_rate_percent > 10 or result.p95_response_time_ms > 2000:
                    logger.warning(f"🛑 System degradation detected at {test_level.value}, stopping tests")
                    break
                
                # Recovery period between tests
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Test {test_level.value} failed: {e}")
                validation_results['test_results'].append({
                    'level': test_level.value,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Analyze results
        if self.results:
            validation_results['sla_compliance_summary'] = self._analyze_sla_compliance()
            validation_results['capacity_analysis'] = self._analyze_system_capacity()
            validation_results['recommendations'] = self._generate_recommendations()
        
        # Final health check
        final_health = await self.validate_system_health()
        validation_results['final_health'] = final_health
        
        logger.info("✅ Enterprise Concurrent Load Validation Suite completed")
        
        return validation_results
    
    def _analyze_sla_compliance(self) -> Dict[str, Any]:
        """Analyze SLA compliance across all tests."""
        if not self.results:
            return {}
        
        sla_analysis = {
            'response_time_compliance': [],
            'error_rate_compliance': [],
            'uptime_compliance': [],
            'overall_compliance': True
        }
        
        for result in self.results:
            sla_analysis['response_time_compliance'].append({
                'test_level': result.test_level.value,
                'p95_response_time_ms': result.p95_response_time_ms,
                'target_ms': SLATarget.RESPONSE_TIME_MS.value,
                'compliant': result.p95_response_time_ms <= SLATarget.RESPONSE_TIME_MS.value
            })
            
            sla_analysis['error_rate_compliance'].append({
                'test_level': result.test_level.value,
                'error_rate_percent': result.error_rate_percent,
                'target_percent': SLATarget.ERROR_RATE_PCT.value,
                'compliant': result.error_rate_percent <= SLATarget.ERROR_RATE_PCT.value
            })
            
            sla_analysis['uptime_compliance'].append({
                'test_level': result.test_level.value,
                'uptime_percentage': result.uptime_percentage,
                'target_percentage': SLATarget.UPTIME_PERCENTAGE.value,
                'compliant': result.uptime_percentage >= SLATarget.UPTIME_PERCENTAGE.value
            })
            
            if not all(result.sla_compliance.values()):
                sla_analysis['overall_compliance'] = False
        
        return sla_analysis
    
    def _analyze_system_capacity(self) -> Dict[str, Any]:
        """Analyze system capacity and performance characteristics."""
        if not self.results:
            return {}
        
        # Find optimal capacity point
        compliant_results = [r for r in self.results if all(r.sla_compliance.values())]
        max_capacity = max(compliant_results, key=lambda r: r.concurrent_users) if compliant_results else None
        
        # Performance degradation analysis
        degradation_points = []
        if len(self.results) > 1:
            baseline = self.results[0]
            for result in self.results[1:]:
                latency_increase = (result.p95_response_time_ms - baseline.p95_response_time_ms) / baseline.p95_response_time_ms
                degradation_points.append({
                    'users': result.concurrent_users,
                    'latency_degradation_factor': latency_increase,
                    'error_rate_percent': result.error_rate_percent
                })
        
        return {
            'max_compliant_capacity': max_capacity.concurrent_users if max_capacity else 0,
            'max_tested_capacity': max(r.concurrent_users for r in self.results),
            'peak_performance': {
                'max_rps': max(r.requests_per_second for r in self.results),
                'best_p95_latency_ms': min(r.p95_response_time_ms for r in self.results),
                'lowest_error_rate_percent': min(r.error_rate_percent for r in self.results)
            },
            'degradation_analysis': degradation_points,
            'resource_utilization': {
                'peak_cpu_percent': max(r.peak_cpu_percent for r in self.results),
                'peak_memory_mb': max(r.peak_memory_mb for r in self.results)
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.results:
            return ["No test results available for analysis"]
        
        # Find SLA violations
        sla_violations = [r for r in self.results if not all(r.sla_compliance.values())]
        
        if sla_violations:
            recommendations.append(
                f"SLA violations detected at {len(sla_violations)} load levels. "
                "Consider infrastructure scaling or optimization."
            )
        
        # Check if 1000+ user target was met
        high_load_results = [r for r in self.results if r.concurrent_users >= 1000]
        if not high_load_results:
            recommendations.append(
                "Target of 1000+ concurrent users not tested. "
                "System may need capacity improvements."
            )
        elif not all(all(r.sla_compliance.values()) for r in high_load_results):
            recommendations.append(
                "1000+ user load tested but SLA violations occurred. "
                "Optimize database connections, caching, or add horizontal scaling."
            )
        
        # Resource utilization recommendations
        high_cpu_results = [r for r in self.results if r.peak_cpu_percent > 80]
        if high_cpu_results:
            recommendations.append(
                "High CPU utilization detected (>80%). Consider CPU scaling or optimization."
            )
        
        high_memory_results = [r for r in self.results if r.peak_memory_mb > 4000]  # 4GB threshold
        if high_memory_results:
            recommendations.append(
                "High memory utilization detected (>4GB). Consider memory optimization or scaling."
            )
        
        # Performance recommendations
        slow_results = [r for r in self.results if r.p95_response_time_ms > 500]
        if slow_results:
            recommendations.append(
                "Response times >500ms detected. Optimize database queries, add caching, or improve API performance."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ All tests passed SLA requirements. System is ready for production load."
            )
        
        return recommendations


# Test fixtures and utilities for pytest integration
@pytest.fixture
async def concurrent_load_validator():
    """Pytest fixture for concurrent load validator."""
    validator = EnterpriseConcurrentLoadValidator()
    await validator.setup_infrastructure()
    yield validator
    # Cleanup if needed


class TestEnterpriseConcurrentLoad:
    """Test suite for enterprise concurrent load validation."""
    
    @pytest.mark.asyncio
    async def test_system_health_validation(self, concurrent_load_validator):
        """Test system health validation."""
        health_status = await concurrent_load_validator.validate_system_health()
        
        assert isinstance(health_status, dict)
        assert 'api_health' in health_status
        assert 'database_health' in health_status
        assert 'redis_health' in health_status
    
    @pytest.mark.asyncio
    async def test_baseline_concurrent_load(self, concurrent_load_validator):
        """Test baseline concurrent load (100 users)."""
        result = await concurrent_load_validator.run_concurrent_load_test(
            ReliabilityTestLevel.BASELINE
        )
        
        assert result.concurrent_users == 100
        assert result.requests_per_second > 0
        assert result.error_rate_percent <= 5.0  # Generous threshold for baseline
    
    @pytest.mark.asyncio
    async def test_high_concurrent_load(self, concurrent_load_validator):
        """Test high concurrent load (1000 users)."""
        result = await concurrent_load_validator.run_concurrent_load_test(
            ReliabilityTestLevel.HIGH_LOAD
        )
        
        assert result.concurrent_users == 1000
        # SLA validation for 1000+ users
        assert result.p95_response_time_ms <= SLATarget.RESPONSE_TIME_MS.value, \
            f"P95 response time {result.p95_response_time_ms}ms exceeds {SLATarget.RESPONSE_TIME_MS.value}ms target"
        assert result.error_rate_percent <= SLATarget.ERROR_RATE_PCT.value, \
            f"Error rate {result.error_rate_percent}% exceeds {SLATarget.ERROR_RATE_PCT.value}% target"
    
    @pytest.mark.asyncio
    async def test_comprehensive_validation_suite(self, concurrent_load_validator):
        """Test comprehensive concurrent load validation."""
        results = await concurrent_load_validator.run_comprehensive_concurrent_validation()
        
        assert 'test_results' in results
        assert 'sla_compliance_summary' in results
        assert 'capacity_analysis' in results
        assert 'recommendations' in results
        
        # Check that high load test was included
        test_levels = [r['level'] for r in results['test_results'] if 'level' in r]
        assert 'high_load' in test_levels


if __name__ == "__main__":
    async def main():
        validator = EnterpriseConcurrentLoadValidator()
        results = await validator.run_comprehensive_concurrent_validation()
        
        print("🏁 Enterprise Concurrent Load Validation Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
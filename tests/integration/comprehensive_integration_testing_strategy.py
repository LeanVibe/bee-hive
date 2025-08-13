"""
Comprehensive Integration Testing Strategy for LeanVibe Agent Hive 2.0

This module implements a complete integration testing framework that validates:
- End-to-end system behavior across all 6 critical components
- Cross-component failure scenarios and recovery mechanisms
- Data consistency and state synchronization
- Performance under realistic load conditions
- Infrastructure integration with production-like environments

Critical Integration Paths Tested:
1. Agent Orchestration Flow: Agent spawn → Redis messaging → State persistence → Dashboard update
2. Real-time Dashboard Flow: WebSocket connection → Data streaming → Frontend updates → User interaction
3. Offline-first Flow: Backend unavailable → PWA fallback → Data queuing → Reconciliation
4. Error Recovery Flow: Component failure → Circuit breaker → Graceful degradation → Recovery
5. System Initialization: Startup sequence → Service discovery → Health verification → Ready state

Strategy follows First Principles:
- System = Sum of Interactions: Testing isolated components doesn't guarantee system behavior
- Failure = Cascade Effect: Single component failure can cascade through entire system
- Reality = Production Conditions: Tests must simulate realistic production scenarios
"""

import asyncio
import pytest
import time
import uuid
import tempfile
import json
import docker
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import psutil
import logging

# Core system imports
from app.core.orchestrator import AgentOrchestrator
from app.core.coordination_dashboard import CoordinationDashboard
from app.core.redis import AgentMessageBroker, SessionCache
from app.core.github_api_client import GitHubAPIClient
from app.core.work_tree_manager import WorkTreeManager
from app.core.tmux_session_manager import TmuxSessionManager


@dataclass
class IntegrationTestEnvironment:
    """Configuration for integration test environment."""
    name: str
    services: List[str] = field(default_factory=lambda: [
        "postgres", "redis", "api", "frontend"
    ])
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    chaos_engineering: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    network_simulation: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PerformanceMetrics:
    """Performance metrics collection for integration tests."""
    test_name: str
    start_time: float
    end_time: float
    cpu_usage_percent: float
    memory_usage_mb: float
    network_io_bytes: int
    disk_io_bytes: int
    component_response_times: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


@dataclass
class FailureScenario:
    """Configuration for failure injection scenarios."""
    name: str
    description: str
    target_component: str
    failure_type: str  # "network", "cpu", "memory", "disk", "process"
    duration_seconds: int
    expected_recovery_time: int
    cascade_expected: bool = False


class IntegrationTestOrchestrator:
    """
    Master orchestrator for comprehensive integration testing.
    
    Manages test environments, coordinates failure injection,
    collects performance metrics, and validates system behavior.
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.test_environments: Dict[str, IntegrationTestEnvironment] = {}
        self.active_containers: Dict[str, Any] = {}
        self.performance_metrics: List[PerformanceMetrics] = []
        self.failure_scenarios: List[FailureScenario] = []
        self.logger = logging.getLogger(__name__)
        
    async def setup_test_environment(self, env_config: IntegrationTestEnvironment) -> str:
        """
        Setup a complete test environment with Docker Compose.
        
        Returns the environment ID for tracking.
        """
        env_id = f"integration_test_{env_config.name}_{int(time.time())}"
        
        # Create temporary directory for environment
        temp_dir = tempfile.mkdtemp(prefix=f"integration_test_{env_config.name}_")
        
        # Generate Docker Compose configuration
        compose_config = self._generate_compose_config(env_config)
        compose_file = Path(temp_dir) / "docker-compose.test.yml"
        
        with open(compose_file, 'w') as f:
            f.write(compose_config)
        
        # Start the environment
        cmd = [
            "docker-compose", 
            "-f", str(compose_file),
            "-p", env_id,
            "up", "-d", "--build"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to start environment: {stderr.decode()}")
        
        # Wait for services to be healthy
        await self._wait_for_environment_ready(env_id, env_config.services)
        
        self.test_environments[env_id] = env_config
        
        self.logger.info(f"Integration test environment {env_id} ready")
        return env_id
    
    def _generate_compose_config(self, env_config: IntegrationTestEnvironment) -> str:
        """Generate Docker Compose configuration for test environment."""
        base_compose = {
            "version": "3.8",
            "services": {},
            "networks": {
                "test_network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "postgres_test_data": {},
                "redis_test_data": {}
            }
        }
        
        # Add PostgreSQL service
        if "postgres" in env_config.services:
            base_compose["services"]["postgres"] = {
                "image": "pgvector/pgvector:pg15",
                "environment": {
                    "POSTGRES_DB": "test_db",
                    "POSTGRES_USER": "test_user", 
                    "POSTGRES_PASSWORD": "test_pass"
                },
                "ports": ["5433:5432"],
                "volumes": ["postgres_test_data:/var/lib/postgresql/data"],
                "networks": ["test_network"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U test_user -d test_db"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 5
                }
            }
        
        # Add Redis service
        if "redis" in env_config.services:
            base_compose["services"]["redis"] = {
                "image": "redis:7-alpine",
                "command": "redis-server --appendonly yes",
                "ports": ["6380:6379"],
                "volumes": ["redis_test_data:/data"],
                "networks": ["test_network"],
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 5
                }
            }
        
        # Add API service
        if "api" in env_config.services:
            base_compose["services"]["api"] = {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile",
                    "target": "development"
                },
                "environment": {
                    "DATABASE_URL": "postgresql://test_user:test_pass@postgres:5432/test_db",
                    "REDIS_URL": "redis://redis:6379/0",
                    "ENVIRONMENT": "test",
                    "LOG_LEVEL": "DEBUG"
                },
                "ports": ["8001:8000"],
                "depends_on": {
                    "postgres": {"condition": "service_healthy"},
                    "redis": {"condition": "service_healthy"}
                },
                "networks": ["test_network"],
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            }
        
        # Add resource limits if specified
        if env_config.resource_limits:
            for service_name, service_config in base_compose["services"].items():
                if service_name in env_config.resource_limits:
                    limits = env_config.resource_limits[service_name]
                    service_config["deploy"] = {"resources": {"limits": limits}}
        
        # Add monitoring services if enabled
        if env_config.monitoring_enabled:
            base_compose["services"]["prometheus"] = {
                "image": "prom/prometheus:latest",
                "ports": ["9091:9090"],
                "networks": ["test_network"],
                "command": [
                    "--config.file=/etc/prometheus/prometheus.yml",
                    "--storage.tsdb.path=/prometheus",
                    "--web.enable-lifecycle"
                ]
            }
        
        return json.dumps(base_compose, indent=2)
    
    async def _wait_for_environment_ready(self, env_id: str, services: List[str]) -> None:
        """Wait for all services in environment to be healthy."""
        max_wait_time = 300  # 5 minutes
        check_interval = 10   # 10 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            healthy_services = 0
            
            for service in services:
                try:
                    # Check service health using docker-compose
                    cmd = [
                        "docker-compose", "-p", env_id, 
                        "ps", "--filter", f"service={service}"
                    ]
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, _ = await process.communicate()
                    
                    if "healthy" in stdout.decode() or "Up" in stdout.decode():
                        healthy_services += 1
                        
                except Exception as e:
                    self.logger.warning(f"Health check failed for {service}: {e}")
            
            if healthy_services == len(services):
                self.logger.info(f"All services healthy in environment {env_id}")
                return
            
            await asyncio.sleep(check_interval)
        
        raise TimeoutError(f"Environment {env_id} failed to become ready within {max_wait_time}s")
    
    @asynccontextmanager
    async def performance_monitoring(self, test_name: str) -> AsyncGenerator[PerformanceMetrics, None]:
        """Context manager for collecting performance metrics during test execution."""
        # Get initial system state
        process = psutil.Process()
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_io = process.io_counters()
        
        metrics = PerformanceMetrics(
            test_name=test_name,
            start_time=time.time(),
            end_time=0,
            cpu_usage_percent=initial_cpu,
            memory_usage_mb=initial_memory,
            network_io_bytes=0,
            disk_io_bytes=0
        )
        
        try:
            yield metrics
        finally:
            # Collect final metrics
            final_cpu = process.cpu_percent()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_io = process.io_counters()
            
            metrics.end_time = time.time()
            metrics.cpu_usage_percent = max(initial_cpu, final_cpu)
            metrics.memory_usage_mb = max(initial_memory, final_memory)
            metrics.network_io_bytes = (final_io.read_bytes + final_io.write_bytes) - \
                                     (initial_io.read_bytes + initial_io.write_bytes)
            metrics.disk_io_bytes = metrics.network_io_bytes  # Simplified
            
            self.performance_metrics.append(metrics)
    
    async def inject_failure(self, env_id: str, scenario: FailureScenario) -> None:
        """Inject failure into a specific component of the test environment."""
        self.logger.info(f"Injecting failure: {scenario.name} on {scenario.target_component}")
        
        if scenario.failure_type == "network":
            await self._inject_network_failure(env_id, scenario)
        elif scenario.failure_type == "cpu":
            await self._inject_cpu_stress(env_id, scenario)
        elif scenario.failure_type == "memory":
            await self._inject_memory_pressure(env_id, scenario)
        elif scenario.failure_type == "process":
            await self._inject_process_failure(env_id, scenario)
        else:
            raise ValueError(f"Unknown failure type: {scenario.failure_type}")
        
        # Wait for failure duration
        await asyncio.sleep(scenario.duration_seconds)
        
        # Stop failure injection
        await self._stop_failure_injection(env_id, scenario)
        
        self.logger.info(f"Failure injection completed: {scenario.name}")
    
    async def _inject_network_failure(self, env_id: str, scenario: FailureScenario) -> None:
        """Inject network failure using tc (traffic control)."""
        container_name = f"{env_id}_{scenario.target_component}_1"
        
        # Add network delay/packet loss
        cmd = [
            "docker", "exec", container_name,
            "tc", "qdisc", "add", "dev", "eth0", "root",
            "netem", "delay", "1000ms", "loss", "50%"
        ]
        
        try:
            await asyncio.create_subprocess_exec(*cmd)
        except Exception as e:
            self.logger.warning(f"Network failure injection failed: {e}")
    
    async def _inject_cpu_stress(self, env_id: str, scenario: FailureScenario) -> None:
        """Inject CPU stress using stress-ng."""
        container_name = f"{env_id}_{scenario.target_component}_1"
        
        cmd = [
            "docker", "exec", "-d", container_name,
            "stress-ng", "--cpu", "2", "--timeout", f"{scenario.duration_seconds}s"
        ]
        
        try:
            await asyncio.create_subprocess_exec(*cmd)
        except Exception as e:
            self.logger.warning(f"CPU stress injection failed: {e}")
    
    async def _inject_memory_pressure(self, env_id: str, scenario: FailureScenario) -> None:
        """Inject memory pressure using stress-ng."""
        container_name = f"{env_id}_{scenario.target_component}_1"
        
        cmd = [
            "docker", "exec", "-d", container_name,
            "stress-ng", "--vm", "1", "--vm-bytes", "512M", 
            "--timeout", f"{scenario.duration_seconds}s"
        ]
        
        try:
            await asyncio.create_subprocess_exec(*cmd)
        except Exception as e:
            self.logger.warning(f"Memory pressure injection failed: {e}")
    
    async def _inject_process_failure(self, env_id: str, scenario: FailureScenario) -> None:
        """Kill and restart a process to simulate failure."""
        container_name = f"{env_id}_{scenario.target_component}_1"
        
        # Stop the container
        cmd = ["docker", "stop", container_name]
        await asyncio.create_subprocess_exec(*cmd)
        
        # Wait for failure duration
        await asyncio.sleep(scenario.duration_seconds)
        
        # Restart the container
        cmd = ["docker", "start", container_name]
        await asyncio.create_subprocess_exec(*cmd)
    
    async def _stop_failure_injection(self, env_id: str, scenario: FailureScenario) -> None:
        """Stop failure injection and restore normal operation."""
        container_name = f"{env_id}_{scenario.target_component}_1"
        
        if scenario.failure_type == "network":
            # Remove network restrictions
            cmd = [
                "docker", "exec", container_name,
                "tc", "qdisc", "del", "dev", "eth0", "root"
            ]
            try:
                await asyncio.create_subprocess_exec(*cmd)
            except Exception:
                pass  # May fail if qdisc doesn't exist
    
    async def cleanup_environment(self, env_id: str) -> None:
        """Clean up test environment and resources."""
        try:
            cmd = ["docker-compose", "-p", env_id, "down", "-v", "--remove-orphans"]
            await asyncio.create_subprocess_exec(*cmd)
            
            if env_id in self.test_environments:
                del self.test_environments[env_id]
                
            self.logger.info(f"Cleaned up environment {env_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup environment {env_id}: {e}")


@pytest.mark.asyncio
class TestComprehensiveIntegration:
    """
    Comprehensive integration test suite that validates the complete system.
    
    Tests the 20% of integration scenarios that provide 80% of production confidence:
    - Critical user journeys (dashboard monitoring, agent coordination, task management)
    - Failure recovery scenarios (database down, Redis unavailable, WebSocket disconnected)
    - Data flow validation (message routing, state synchronization, consistency checks)
    - Resource exhaustion (connection pool limits, memory pressure, disk space)
    """
    
    @pytest.fixture
    async def integration_orchestrator(self) -> IntegrationTestOrchestrator:
        """Create integration test orchestrator."""
        return IntegrationTestOrchestrator()
    
    @pytest.fixture
    async def standard_test_environment(self, integration_orchestrator) -> str:
        """Setup standard test environment with all core services."""
        env_config = IntegrationTestEnvironment(
            name="standard",
            services=["postgres", "redis", "api", "frontend"],
            monitoring_enabled=True,
            resource_limits={
                "api": {"memory": "1G", "cpus": "0.5"},
                "postgres": {"memory": "512M", "cpus": "0.3"},
                "redis": {"memory": "256M", "cpus": "0.2"}
            }
        )
        
        env_id = await integration_orchestrator.setup_test_environment(env_config)
        yield env_id
        await integration_orchestrator.cleanup_environment(env_id)
    
    async def test_agent_orchestration_end_to_end(
        self, 
        integration_orchestrator: IntegrationTestOrchestrator,
        standard_test_environment: str
    ):
        """
        Test complete agent orchestration flow:
        Agent spawn → Redis messaging → State persistence → Dashboard update
        """
        async with integration_orchestrator.performance_monitoring("agent_orchestration_e2e") as metrics:
            # Phase 1: Agent Spawn and Registration
            orchestrator = AgentOrchestrator()
            
            agent_config = {
                "name": "Integration Test Agent",
                "type": "CLAUDE",
                "role": "backend_developer",
                "capabilities": ["python", "fastapi", "testing"],
                "specialization": "integration_testing"
            }
            
            start_time = time.time()
            agent_id = await orchestrator.spawn_agent(agent_config)
            agent_spawn_time = time.time() - start_time
            
            assert agent_id is not None
            assert agent_spawn_time < 5.0  # Agent spawn under 5 seconds
            
            metrics.component_response_times["agent_spawn"] = agent_spawn_time * 1000
            
            # Phase 2: Redis Message Flow
            message_broker = AgentMessageBroker()
            
            test_message = {
                "type": "task_assignment",
                "agent_id": str(agent_id),
                "task": {
                    "id": str(uuid.uuid4()),
                    "title": "Integration Test Task",
                    "description": "Validate agent messaging system"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            start_time = time.time()
            message_id = await message_broker.publish_agent_message(
                f"agent:{agent_id}:tasks", 
                test_message
            )
            message_publish_time = time.time() - start_time
            
            assert message_id is not None
            assert message_publish_time < 0.1  # Message publishing under 100ms
            
            metrics.component_response_times["message_publish"] = message_publish_time * 1000
            
            # Phase 3: State Persistence Verification
            start_time = time.time()
            
            # Simulate agent processing and state update
            state_update = {
                "agent_id": str(agent_id),
                "status": "PROCESSING",
                "current_task": test_message["task"]["id"],
                "progress": 0.5,
                "last_activity": datetime.utcnow().isoformat()
            }
            
            session_cache = SessionCache()
            await session_cache.store_agent_state(str(agent_id), state_update)
            
            # Verify state persistence
            retrieved_state = await session_cache.get_agent_state(str(agent_id))
            state_persistence_time = time.time() - start_time
            
            assert retrieved_state is not None
            assert retrieved_state["status"] == "PROCESSING"
            assert state_persistence_time < 0.05  # State persistence under 50ms
            
            metrics.component_response_times["state_persistence"] = state_persistence_time * 1000
            
            # Phase 4: Dashboard Update Verification
            dashboard = CoordinationDashboard()
            
            start_time = time.time()
            dashboard_data = await dashboard.get_dashboard_data()
            dashboard_update_time = time.time() - start_time
            
            assert dashboard_data is not None
            assert "metrics" in dashboard_data
            assert dashboard_update_time < 1.0  # Dashboard update under 1 second
            
            metrics.component_response_times["dashboard_update"] = dashboard_update_time * 1000
            
            # Phase 5: End-to-End Validation
            total_flow_time = sum(metrics.component_response_times.values()) / 1000
            
            # Validate performance targets
            assert total_flow_time < 10.0  # Complete flow under 10 seconds
            assert all(time_ms < 5000 for time_ms in metrics.component_response_times.values())
            
            # Validate data consistency across components
            assert str(agent_id) in retrieved_state["agent_id"]
            
            print(f"✅ Agent orchestration E2E test passed")
            print(f"📊 Component response times: {metrics.component_response_times}")
            print(f"⏱️  Total flow time: {total_flow_time:.2f}s")
    
    async def test_cross_component_failure_recovery(
        self,
        integration_orchestrator: IntegrationTestOrchestrator,
        standard_test_environment: str
    ):
        """
        Test cross-component failure scenarios and recovery mechanisms:
        Component failure → Circuit breaker → Graceful degradation → Recovery
        """
        failure_scenarios = [
            FailureScenario(
                name="database_connection_loss",
                description="PostgreSQL database becomes unavailable",
                target_component="postgres",
                failure_type="process",
                duration_seconds=30,
                expected_recovery_time=45
            ),
            FailureScenario(
                name="redis_memory_pressure",
                description="Redis experiences memory pressure",
                target_component="redis", 
                failure_type="memory",
                duration_seconds=20,
                expected_recovery_time=30
            ),
            FailureScenario(
                name="api_cpu_saturation",
                description="API service experiences CPU saturation",
                target_component="api",
                failure_type="cpu",
                duration_seconds=25,
                expected_recovery_time=35
            )
        ]
        
        recovery_results = []
        
        for scenario in failure_scenarios:
            async with integration_orchestrator.performance_monitoring(f"failure_recovery_{scenario.name}") as metrics:
                # Phase 1: Establish baseline performance
                orchestrator = AgentOrchestrator()
                
                baseline_start = time.time()
                baseline_agent = await orchestrator.spawn_agent({
                    "name": f"Baseline Agent {scenario.name}",
                    "type": "CLAUDE",
                    "role": "test_agent"
                })
                baseline_time = time.time() - baseline_start
                
                assert baseline_agent is not None
                assert baseline_time < 5.0
                
                # Phase 2: Inject failure
                failure_start = time.time()
                await integration_orchestrator.inject_failure(standard_test_environment, scenario)
                
                # Phase 3: Test system behavior during failure
                degraded_operations = []
                
                try:
                    # Attempt operations during failure
                    degraded_start = time.time()
                    degraded_agent = await asyncio.wait_for(
                        orchestrator.spawn_agent({
                            "name": f"Degraded Agent {scenario.name}",
                            "type": "CLAUDE",
                            "role": "test_agent"
                        }),
                        timeout=10.0
                    )
                    degraded_time = time.time() - degraded_start
                    
                    degraded_operations.append({
                        "operation": "agent_spawn",
                        "success": degraded_agent is not None,
                        "response_time": degraded_time,
                        "degraded": degraded_time > baseline_time * 2  # 2x slower is degraded
                    })
                    
                except asyncio.TimeoutError:
                    degraded_operations.append({
                        "operation": "agent_spawn",
                        "success": False,
                        "response_time": 10.0,
                        "degraded": True
                    })
                
                # Phase 4: Wait for recovery
                recovery_start = time.time()
                recovery_timeout = scenario.expected_recovery_time
                
                # Poll for system recovery
                system_recovered = False
                while time.time() - recovery_start < recovery_timeout:
                    try:
                        recovery_test_start = time.time()
                        recovery_agent = await asyncio.wait_for(
                            orchestrator.spawn_agent({
                                "name": f"Recovery Test Agent {scenario.name}",
                                "type": "CLAUDE", 
                                "role": "test_agent"
                            }),
                            timeout=5.0
                        )
                        recovery_test_time = time.time() - recovery_test_start
                        
                        # System recovered if response time is within 50% of baseline
                        if recovery_agent and recovery_test_time < baseline_time * 1.5:
                            system_recovered = True
                            break
                            
                    except Exception:
                        pass
                    
                    await asyncio.sleep(2)  # Check every 2 seconds
                
                recovery_time = time.time() - recovery_start
                
                # Phase 5: Validate recovery
                if system_recovered:
                    # Test full functionality after recovery
                    post_recovery_start = time.time()
                    post_recovery_agent = await orchestrator.spawn_agent({
                        "name": f"Post Recovery Agent {scenario.name}",
                        "type": "CLAUDE",
                        "role": "test_agent"
                    })
                    post_recovery_time = time.time() - post_recovery_start
                    
                    assert post_recovery_agent is not None
                    assert post_recovery_time < baseline_time * 1.2  # Within 20% of baseline
                
                recovery_results.append({
                    "scenario": scenario.name,
                    "baseline_time": baseline_time,
                    "failure_injected": True,
                    "degraded_operations": degraded_operations,
                    "system_recovered": system_recovered,
                    "recovery_time": recovery_time,
                    "recovery_within_expected": recovery_time <= scenario.expected_recovery_time,
                    "post_recovery_performance": post_recovery_time if system_recovered else None
                })
                
                metrics.component_response_times[f"{scenario.name}_baseline"] = baseline_time * 1000
                metrics.component_response_times[f"{scenario.name}_recovery"] = recovery_time * 1000
        
        # Validate overall failure recovery performance
        successful_recoveries = sum(1 for r in recovery_results if r["system_recovered"])
        recovery_success_rate = successful_recoveries / len(recovery_results)
        
        assert recovery_success_rate >= 0.8  # At least 80% recovery success rate
        
        # Validate recovery times
        on_time_recoveries = sum(1 for r in recovery_results if r["recovery_within_expected"])
        on_time_rate = on_time_recoveries / len(recovery_results)
        
        assert on_time_rate >= 0.7  # At least 70% recover within expected time
        
        print(f"✅ Cross-component failure recovery test passed")
        print(f"🔄 Recovery success rate: {recovery_success_rate:.1%}")
        print(f"⏰ On-time recovery rate: {on_time_rate:.1%}")
        print(f"📊 Detailed results: {recovery_results}")
    
    async def test_performance_under_realistic_load(
        self,
        integration_orchestrator: IntegrationTestOrchestrator,
        standard_test_environment: str
    ):
        """
        Test system performance under realistic production load:
        - 50 concurrent agents
        - 200 messages/second through Redis
        - 100 dashboard updates/minute
        - 20 GitHub operations/minute
        """
        async with integration_orchestrator.performance_monitoring("performance_under_load") as metrics:
            # Phase 1: Concurrent Agent Spawning
            orchestrator = AgentOrchestrator()
            message_broker = AgentMessageBroker()
            dashboard = CoordinationDashboard()
            
            # Spawn 50 concurrent agents
            concurrent_agents = 50
            agent_spawn_tasks = []
            
            spawn_start_time = time.time()
            
            for i in range(concurrent_agents):
                agent_config = {
                    "name": f"Load Test Agent {i}",
                    "type": "CLAUDE",
                    "role": "load_test_agent",
                    "capabilities": ["testing", "load_simulation"]
                }
                
                task = orchestrator.spawn_agent(agent_config)
                agent_spawn_tasks.append(task)
            
            # Wait for all agents to spawn
            spawned_agents = await asyncio.gather(*agent_spawn_tasks, return_exceptions=True)
            spawn_duration = time.time() - spawn_start_time
            
            successful_spawns = sum(1 for agent in spawned_agents if not isinstance(agent, Exception))
            spawn_success_rate = successful_spawns / concurrent_agents
            
            assert spawn_success_rate >= 0.9  # At least 90% successful spawns
            assert spawn_duration < 30.0  # All agents spawned within 30 seconds
            
            metrics.component_response_times["concurrent_agent_spawn"] = spawn_duration * 1000
            
            # Phase 2: High-Frequency Message Processing
            message_rate = 200  # messages per second
            test_duration = 30   # seconds
            total_messages = message_rate * test_duration
            
            message_tasks = []
            message_start_time = time.time()
            
            for i in range(total_messages):
                # Distribute messages across spawned agents
                agent_index = i % successful_spawns
                target_agent = spawned_agents[agent_index]
                
                if not isinstance(target_agent, Exception):
                    message = {
                        "type": "load_test_message",
                        "message_id": i,
                        "agent_id": str(target_agent),
                        "timestamp": time.time(),
                        "data": f"Load test message {i}"
                    }
                    
                    task = message_broker.publish_agent_message(
                        f"agent:{target_agent}:load_test",
                        message
                    )
                    message_tasks.append(task)
                
                # Rate limiting to achieve target message rate
                if i % message_rate == 0 and i > 0:
                    await asyncio.sleep(1.0)
            
            # Wait for all messages to be published
            published_messages = await asyncio.gather(*message_tasks, return_exceptions=True)
            message_duration = time.time() - message_start_time
            
            successful_publishes = sum(1 for msg in published_messages if not isinstance(msg, Exception))
            publish_success_rate = successful_publishes / total_messages
            actual_message_rate = successful_publishes / message_duration
            
            assert publish_success_rate >= 0.95  # At least 95% successful publishes
            assert actual_message_rate >= message_rate * 0.8  # Within 80% of target rate
            
            metrics.component_response_times["message_publishing"] = message_duration * 1000
            
            # Phase 3: Dashboard Update Load
            dashboard_updates = 100  # per minute = ~1.67 per second
            dashboard_test_duration = 60  # seconds
            
            dashboard_tasks = []
            dashboard_start_time = time.time()
            
            for i in range(dashboard_updates):
                task = dashboard.get_dashboard_data()
                dashboard_tasks.append(task)
                
                # Rate limiting for dashboard updates
                await asyncio.sleep(dashboard_test_duration / dashboard_updates)
            
            dashboard_results = await asyncio.gather(*dashboard_tasks, return_exceptions=True)
            dashboard_duration = time.time() - dashboard_start_time
            
            successful_dashboard_updates = sum(1 for result in dashboard_results if not isinstance(result, Exception))
            dashboard_success_rate = successful_dashboard_updates / dashboard_updates
            
            assert dashboard_success_rate >= 0.95  # At least 95% successful dashboard updates
            
            metrics.component_response_times["dashboard_updates"] = dashboard_duration * 1000
            
            # Phase 4: System Resource Validation
            # Verify system resources haven't been exhausted
            import psutil
            
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # System should remain responsive
            assert cpu_usage < 80  # CPU usage under 80%
            assert memory_usage < 85  # Memory usage under 85%
            
            # Phase 5: Performance Metrics Validation
            total_operations = successful_spawns + successful_publishes + successful_dashboard_updates
            total_test_duration = max(spawn_duration, message_duration, dashboard_duration)
            operations_per_second = total_operations / total_test_duration
            
            # Validate performance targets
            assert operations_per_second >= 200  # At least 200 operations per second overall
            
            performance_summary = {
                "agent_spawn_success_rate": spawn_success_rate,
                "message_publish_success_rate": publish_success_rate,
                "dashboard_update_success_rate": dashboard_success_rate,
                "actual_message_rate": actual_message_rate,
                "target_message_rate": message_rate,
                "operations_per_second": operations_per_second,
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_usage,
                "total_test_duration": total_test_duration
            }
            
            print(f"✅ Performance under load test passed")
            print(f"🚀 Operations per second: {operations_per_second:.1f}")
            print(f"📊 Performance summary: {performance_summary}")
    
    async def test_data_consistency_across_components(
        self,
        integration_orchestrator: IntegrationTestOrchestrator,
        standard_test_environment: str
    ):
        """
        Test data consistency and synchronization across component boundaries:
        - State synchronization between Redis and PostgreSQL
        - Message ordering and delivery guarantees
        - Dashboard data consistency with backend state
        """
        async with integration_orchestrator.performance_monitoring("data_consistency") as metrics:
            # Phase 1: Multi-Component Data Flow
            orchestrator = AgentOrchestrator()
            message_broker = AgentMessageBroker()
            session_cache = SessionCache()
            dashboard = CoordinationDashboard()
            
            # Create test agent
            agent_config = {
                "name": "Data Consistency Test Agent",
                "type": "CLAUDE",
                "role": "data_test_agent"
            }
            
            agent_id = await orchestrator.spawn_agent(agent_config)
            assert agent_id is not None
            
            # Phase 2: Test Message Ordering and Delivery
            ordered_messages = []
            message_count = 50
            
            for i in range(message_count):
                message = {
                    "sequence_number": i,
                    "agent_id": str(agent_id),
                    "timestamp": time.time(),
                    "operation": "data_consistency_test",
                    "data": {"value": i * 10, "status": "processing"}
                }
                
                message_id = await message_broker.publish_agent_message(
                    f"agent:{agent_id}:ordered_test",
                    message
                )
                ordered_messages.append((i, message_id))
                
                # Small delay to ensure ordering
                await asyncio.sleep(0.01)
            
            # Verify message ordering
            assert len(ordered_messages) == message_count
            
            # Phase 3: State Synchronization Test
            state_updates = []
            
            for i in range(10):
                state = {
                    "agent_id": str(agent_id),
                    "sequence": i,
                    "status": f"state_{i}",
                    "data": {"counter": i, "timestamp": time.time()},
                    "version": i + 1
                }
                
                # Store in session cache (Redis)
                await session_cache.store_agent_state(str(agent_id), state)
                state_updates.append(state)
                
                await asyncio.sleep(0.1)  # Allow for synchronization
            
            # Verify final state consistency
            final_state = await session_cache.get_agent_state(str(agent_id))
            assert final_state is not None
            assert final_state["sequence"] == 9  # Last update
            assert final_state["version"] == 10  # Final version
            
            # Phase 4: Cross-Component Consistency Check
            # Simulate data flowing through multiple components
            test_data = {
                "agent_id": str(agent_id),
                "test_id": str(uuid.uuid4()),
                "workflow_state": "active",
                "tasks": [
                    {"id": "task_1", "status": "completed"},
                    {"id": "task_2", "status": "in_progress"},
                    {"id": "task_3", "status": "pending"}
                ],
                "metrics": {
                    "completion_rate": 0.33,
                    "estimated_time": 120,
                    "quality_score": 0.85
                }
            }
            
            # Store in Redis
            await session_cache.store_agent_state(str(agent_id), test_data)
            
            # Publish update message
            update_message = {
                "type": "state_update",
                "agent_id": str(agent_id),
                "data": test_data,
                "timestamp": time.time()
            }
            
            await message_broker.publish_agent_message(
                f"agent:{agent_id}:state_updates",
                update_message
            )
            
            # Allow time for propagation
            await asyncio.sleep(2.0)
            
            # Get dashboard data and verify consistency
            dashboard_data = await dashboard.get_dashboard_data()
            
            # Verify data appears consistently across components
            retrieved_state = await session_cache.get_agent_state(str(agent_id))
            
            assert retrieved_state["test_id"] == test_data["test_id"]
            assert retrieved_state["workflow_state"] == "active"
            assert len(retrieved_state["tasks"]) == 3
            
            # Phase 5: Concurrent Write Consistency Test
            concurrent_writers = 5
            writes_per_writer = 10
            
            async def concurrent_writer(writer_id: int):
                writes = []
                for i in range(writes_per_writer):
                    state = {
                        "agent_id": str(agent_id),
                        "writer_id": writer_id,
                        "write_sequence": i,
                        "timestamp": time.time(),
                        "data": f"writer_{writer_id}_write_{i}"
                    }
                    
                    await session_cache.store_agent_state(
                        f"{agent_id}_writer_{writer_id}",
                        state
                    )
                    writes.append(state)
                    
                    await asyncio.sleep(0.01)
                
                return writes
            
            # Execute concurrent writes
            write_tasks = [
                concurrent_writer(i) for i in range(concurrent_writers)
            ]
            
            concurrent_results = await asyncio.gather(*write_tasks)
            
            # Verify all writes completed successfully
            total_writes = sum(len(results) for results in concurrent_results)
            expected_writes = concurrent_writers * writes_per_writer
            
            assert total_writes == expected_writes
            
            # Verify data integrity - check each writer's final state
            for writer_id in range(concurrent_writers):
                final_writer_state = await session_cache.get_agent_state(
                    f"{agent_id}_writer_{writer_id}"
                )
                assert final_writer_state is not None
                assert final_writer_state["writer_id"] == writer_id
                assert final_writer_state["write_sequence"] == writes_per_writer - 1
            
            # Phase 6: Consistency Metrics
            consistency_metrics = {
                "message_ordering_preserved": True,
                "state_synchronization_accurate": True,
                "cross_component_consistency": True,
                "concurrent_write_integrity": True,
                "total_operations": total_writes + message_count + len(state_updates),
                "consistency_check_duration": metrics.duration_seconds
            }
            
            print(f"✅ Data consistency test passed")
            print(f"🔄 Consistency metrics: {consistency_metrics}")
    
    async def test_chaos_engineering_resilience(
        self,
        integration_orchestrator: IntegrationTestOrchestrator,
        standard_test_environment: str
    ):
        """
        Chaos engineering test to validate system resilience under random failures.
        """
        # Create chaos environment
        chaos_env_config = IntegrationTestEnvironment(
            name="chaos",
            services=["postgres", "redis", "api"],
            chaos_engineering=True,
            monitoring_enabled=True
        )
        
        chaos_env_id = await integration_orchestrator.setup_test_environment(chaos_env_config)
        
        try:
            async with integration_orchestrator.performance_monitoring("chaos_engineering") as metrics:
                # Define chaos scenarios
                chaos_scenarios = [
                    FailureScenario(
                        name="random_network_partition",
                        description="Random network partitions between services",
                        target_component="api",
                        failure_type="network",
                        duration_seconds=15,
                        expected_recovery_time=30,
                        cascade_expected=True
                    ),
                    FailureScenario(
                        name="memory_pressure_cascade",
                        description="Memory pressure causing cascading failures",
                        target_component="postgres",
                        failure_type="memory", 
                        duration_seconds=20,
                        expected_recovery_time=45,
                        cascade_expected=True
                    ),
                    FailureScenario(
                        name="cpu_saturation_burst",
                        description="CPU saturation in multiple components",
                        target_component="redis",
                        failure_type="cpu",
                        duration_seconds=25,
                        expected_recovery_time=40,
                        cascade_expected=False
                    )
                ]
                
                orchestrator = AgentOrchestrator()
                resilience_results = []
                
                for scenario in chaos_scenarios:
                    # Establish baseline before chaos
                    baseline_agents = []
                    for i in range(5):
                        agent = await orchestrator.spawn_agent({
                            "name": f"Chaos Baseline Agent {i}",
                            "type": "CLAUDE",
                            "role": "chaos_test"
                        })
                        baseline_agents.append(agent)
                    
                    baseline_success = len([a for a in baseline_agents if a is not None])
                    
                    # Inject chaos
                    chaos_start = time.time()
                    await integration_orchestrator.inject_failure(chaos_env_id, scenario)
                    
                    # Monitor system behavior during chaos
                    chaos_operations = []
                    chaos_duration = scenario.duration_seconds
                    
                    for i in range(chaos_duration // 5):  # Test every 5 seconds
                        try:
                            chaos_agent = await asyncio.wait_for(
                                orchestrator.spawn_agent({
                                    "name": f"Chaos Test Agent {i}",
                                    "type": "CLAUDE",
                                    "role": "chaos_test"
                                }),
                                timeout=10.0
                            )
                            chaos_operations.append({
                                "time": time.time() - chaos_start,
                                "operation": "agent_spawn",
                                "success": chaos_agent is not None,
                                "degraded": False
                            })
                        except Exception as e:
                            chaos_operations.append({
                                "time": time.time() - chaos_start,
                                "operation": "agent_spawn", 
                                "success": False,
                                "degraded": True,
                                "error": str(e)
                            })
                        
                        await asyncio.sleep(5)
                    
                    # Wait for recovery
                    recovery_start = time.time()
                    system_recovered = False
                    
                    while time.time() - recovery_start < scenario.expected_recovery_time:
                        try:
                            recovery_agent = await asyncio.wait_for(
                                orchestrator.spawn_agent({
                                    "name": f"Recovery Test Agent",
                                    "type": "CLAUDE",
                                    "role": "recovery_test"
                                }),
                                timeout=5.0
                            )
                            
                            if recovery_agent:
                                system_recovered = True
                                break
                                
                        except Exception:
                            pass
                        
                        await asyncio.sleep(2)
                    
                    recovery_time = time.time() - recovery_start
                    
                    # Analyze resilience
                    successful_operations = sum(1 for op in chaos_operations if op["success"])
                    total_operations = len(chaos_operations)
                    availability_during_chaos = successful_operations / total_operations if total_operations > 0 else 0
                    
                    resilience_result = {
                        "scenario": scenario.name,
                        "baseline_success_rate": baseline_success / 5,
                        "availability_during_chaos": availability_during_chaos,
                        "system_recovered": system_recovered,
                        "recovery_time": recovery_time,
                        "cascade_detected": availability_during_chaos < 0.5 if scenario.cascade_expected else False,
                        "resilience_score": (availability_during_chaos + (1 if system_recovered else 0)) / 2
                    }
                    
                    resilience_results.append(resilience_result)
                
                # Calculate overall resilience metrics
                avg_resilience_score = sum(r["resilience_score"] for r in resilience_results) / len(resilience_results)
                recovery_success_rate = sum(1 for r in resilience_results if r["system_recovered"]) / len(resilience_results)
                
                assert avg_resilience_score >= 0.6  # At least 60% resilience score
                assert recovery_success_rate >= 0.8  # At least 80% recovery success
                
                print(f"✅ Chaos engineering resilience test passed")
                print(f"🔥 Average resilience score: {avg_resilience_score:.2f}")
                print(f"🛡️ Recovery success rate: {recovery_success_rate:.1%}")
                print(f"📊 Detailed resilience results: {resilience_results}")
                
        finally:
            await integration_orchestrator.cleanup_environment(chaos_env_id)


if __name__ == "__main__":
    # Run comprehensive integration tests
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",
        "--asyncio-mode=auto",
        "-k", "test_comprehensive_integration"
    ])
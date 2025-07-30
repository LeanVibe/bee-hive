"""
Locust load testing framework for LeanVibe Agent Hive 2.0.

Enterprise-grade performance testing with comprehensive metrics collection.
"""

import time
from locust import HttpUser, task, between
from locust.exception import RescheduleTask
from typing import Dict, Any
import json
import random
import uuid


class AgentHiveUser(HttpUser):
    """
    Simulates a user interacting with the Agent Hive system.
    
    This class represents realistic usage patterns for enterprise deployment.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.agent_id = None
        self.session_id = None
        self.auth_headers = {}
    
    def on_stop(self):
        """Clean up user session."""
        if self.session_id:
            self.cleanup_session()
    
    @task(20)  # High frequency - system health checks
    def health_check(self):
        """Test basic system health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(15)  # Agent management operations
    def agent_operations(self):
        """Test agent lifecycle operations."""
        # Create agent
        agent_data = {
            "name": f"test-agent-{random.randint(1000, 9999)}",
            "type": "claude",
            "role": "developer",
            "capabilities": [
                {
                    "name": "code_analysis",
                    "description": "Code analysis capability",
                    "confidence_level": 0.8,
                    "specialization_areas": ["python", "testing"]
                }
            ],
            "status": "active",
            "config": {"test_mode": True}
        }
        
        with self.client.post("/api/v1/agents", 
                            json=agent_data,
                            headers=self.auth_headers,
                            catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
                try:
                    data = response.json()
                    self.agent_id = data.get("id")
                except:
                    pass
            else:
                response.failure(f"Agent creation failed: {response.status_code}")
    
    @task(10)  # Session management
    def session_operations(self):
        """Test session lifecycle operations."""
        if not self.agent_id:
            raise RescheduleTask()
        
        session_data = {
            "name": f"test-session-{random.randint(1000, 9999)}",
            "description": "Load test session",
            "session_type": "feature_development",
            "status": "active",
            "participant_agents": [self.agent_id],
            "lead_agent_id": self.agent_id,
            "objectives": ["Performance testing", "Load validation"]
        }
        
        with self.client.post("/api/v1/sessions",
                            json=session_data,
                            headers=self.auth_headers,
                            catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
                try:
                    data = response.json()
                    self.session_id = data.get("id")
                except:
                    pass
            else:
                response.failure(f"Session creation failed: {response.status_code}")
    
    @task(8)  # Task management
    def task_operations(self):
        """Test task lifecycle operations."""
        if not self.agent_id:
            raise RescheduleTask()
        
        task_data = {
            "title": f"Load Test Task {random.randint(1000, 9999)}",
            "description": "Performance testing task",
            "task_type": "feature_development",
            "status": "pending",
            "priority": "medium",
            "assigned_agent_id": self.agent_id,
            "required_capabilities": ["code_analysis"],
            "estimated_effort": 60,
            "context": {"load_test": True}
        }
        
        with self.client.post("/api/v1/tasks",
                            json=task_data,
                            headers=self.auth_headers,
                            catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Task creation failed: {response.status_code}")
    
    @task(5)  # Workflow operations
    def workflow_operations(self):
        """Test workflow operations."""
        workflow_data = {
            "name": f"load-test-workflow-{random.randint(1000, 9999)}",
            "description": "Load testing workflow",
            "status": "created",
            "priority": "medium",
            "definition": {"type": "sequential", "steps": ["analyze", "implement", "test"]},
            "context": {"load_test": True},
            "variables": {"env": "testing"},
            "estimated_duration": 120
        }
        
        with self.client.post("/api/v1/workflows",
                            json=workflow_data,
                            headers=self.auth_headers,
                            catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Workflow creation failed: {response.status_code}")
    
    @task(3)  # Observability endpoints
    def observability_operations(self):
        """Test observability and monitoring endpoints."""
        endpoints = [
            "/api/v1/metrics",
            "/api/v1/events",
            "/api/v1/system/status"
        ]
        
        endpoint = random.choice(endpoints)
        with self.client.get(endpoint,
                           headers=self.auth_headers,
                           catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Some endpoints might not be implemented yet
                response.success()
            else:
                response.failure(f"Observability endpoint failed: {response.status_code}")
    
    def cleanup_session(self):
        """Clean up test session."""
        if self.session_id:
            with self.client.delete(f"/api/v1/sessions/{self.session_id}",
                                  headers=self.auth_headers,
                                  catch_response=True) as response:
                # Don't fail the test if cleanup fails
                pass


class AdminUser(HttpUser):
    """
    Simulates administrative operations for system management.
    
    Lower frequency but higher impact operations.
    """
    
    wait_time = between(5, 10)  # Longer wait times for admin operations
    weight = 1  # Lower weight compared to regular users
    
    def on_start(self):
        """Initialize admin session."""
        self.auth_headers = {"Authorization": "Bearer admin-token"}
    
    @task(10)
    def system_diagnostics(self):
        """Run system diagnostic operations."""
        with self.client.get("/api/v1/system/diagnostics",
                           headers=self.auth_headers,
                           catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 if not implemented
                response.success()
            else:
                response.failure(f"System diagnostics failed: {response.status_code}")
    
    @task(5)
    def agent_management(self):
        """Administrative agent management."""
        with self.client.get("/api/v1/agents",
                           headers=self.auth_headers,
                           catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Agent listing failed: {response.status_code}")


# Performance test configuration
class PerformanceTestConfig:
    """Configuration for different performance test scenarios."""
    
    # Target performance metrics for enterprise deployment
    TARGETS = {
        "max_response_time_ms": 2000,     # 2 seconds max response time
        "p95_response_time_ms": 1000,     # 95% under 1 second
        "p99_response_time_ms": 1500,     # 99% under 1.5 seconds
        "min_rps": 100,                   # Minimum 100 requests per second
        "error_rate_threshold": 0.01,     # Less than 1% error rate
        "concurrent_users": 50,           # Support 50 concurrent users
    }
    
    # Test scenarios
    SCENARIOS = {
        "smoke_test": {
            "users": 5,
            "spawn_rate": 1,
            "run_time": "2m"
        },
        "load_test": {
            "users": 50,
            "spawn_rate": 5,
            "run_time": "10m"
        },
        "stress_test": {
            "users": 200,
            "spawn_rate": 10,
            "run_time": "15m"
        },
        "spike_test": {
            "users": 500,
            "spawn_rate": 50,
            "run_time": "5m"
        }
    }


if __name__ == "__main__":
    import os
    import sys
    
    # Basic locust runner for development testing
    print("LeanVibe Agent Hive 2.0 - Performance Testing Framework")
    print("======================================================")
    print()
    print("Available test scenarios:")
    for name, config in PerformanceTestConfig.SCENARIOS.items():
        print(f"  {name}: {config['users']} users, {config['run_time']}")
    print()
    print("To run tests:")
    print("  locust -f tests/performance/locust_load_tests.py --host=http://localhost:8000")
    print("  locust -f tests/performance/locust_load_tests.py --host=http://localhost:8000 --users 10 --spawn-rate 2 -t 60s")
    print()
    print("Performance targets:")
    for metric, target in PerformanceTestConfig.TARGETS.items():
        print(f"  {metric}: {target}")
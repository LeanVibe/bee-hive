"""
Comprehensive Test Suite for Enhanced /hive Commands
Tests enhanced command performance, mobile optimization, and intelligent caching.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from app.api.hive_commands import (
    HiveCommandRequest, 
    HiveCommandResponse, 
    execute_command,
    _is_cacheable_command,
    _get_cache_ttl
)
from app.core.hive_slash_commands import (
    HiveSlashCommandRegistry,
    HiveStatusCommand,
    HiveFocusCommand,
    HiveProductivityCommand,
    execute_hive_command
)


class TestEnhancedHiveCommandPerformance:
    """Test performance targets for enhanced /hive commands."""
    
    @pytest.mark.asyncio
    async def test_mobile_optimized_status_response_time(self):
        """Test mobile-optimized status command meets <5ms cache target."""
        # Setup mobile-optimized request
        request = HiveCommandRequest(
            command="/hive:status --mobile --priority=high",
            mobile_optimized=True,
            use_cache=True,
            priority="high"
        )
        
        # Mock cache hit scenario
        with patch('app.api.hive_commands.get_cached_mobile_response') as mock_cache:
            mock_cache.return_value = {
                "success": True,
                "mobile_optimized": True,
                "system_state": "operational",
                "agent_count": 5,
                "quick_actions": []
            }
            
            start_time = time.time()
            response = await execute_command(request)
            execution_time = (time.time() - start_time) * 1000
            
            # Validate response time target
            assert execution_time < 5.0, f"Mobile cache response took {execution_time}ms, target <5ms"
            assert response.success
            assert response.mobile_optimized
            assert response.cached
            assert response.execution_time_ms < 5.0
    
    @pytest.mark.asyncio
    async def test_live_command_response_time_target(self):
        """Test live command execution meets <50ms mobile target."""
        request = HiveCommandRequest(
            command="/hive:status --mobile",
            mobile_optimized=True,
            use_cache=False  # Force live execution
        )
        
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
            mock_agents.return_value = {"agent1": {"role": "backend", "status": "active"}}
            
            start_time = time.time()
            response = await execute_command(request)
            execution_time = (time.time() - start_time) * 1000
            
            # Validate mobile live response target
            assert execution_time < 50.0, f"Mobile live response took {execution_time}ms, target <50ms"
            assert response.success
            assert response.mobile_optimized
            assert not response.cached
    
    @pytest.mark.asyncio
    async def test_intelligent_cache_effectiveness(self):
        """Test intelligent caching system effectiveness."""
        # Test cacheable commands
        cacheable_commands = [
            "/hive:status --mobile",
            "/hive:focus development", 
            "/hive:productivity --mobile",
            "/hive:notifications --summary"
        ]
        
        for command in cacheable_commands:
            assert _is_cacheable_command(command), f"Command {command} should be cacheable"
        
        # Test non-cacheable commands
        non_cacheable_commands = [
            "/hive:start --team-size=5",
            "/hive:spawn backend_developer",
            "/hive:develop \"Build API\"",
            "/hive:stop --force"
        ]
        
        for command in non_cacheable_commands:
            assert not _is_cacheable_command(command), f"Command {command} should not be cacheable"
    
    def test_cache_ttl_strategy(self):
        """Test cache TTL strategy based on command type and priority."""
        # Critical priority should have shorter TTL
        critical_ttl = _get_cache_ttl("/hive:status", "critical")
        medium_ttl = _get_cache_ttl("/hive:status", "medium")
        assert critical_ttl < medium_ttl, "Critical priority should have shorter TTL"
        
        # Different command types should have appropriate TTLs
        status_ttl = _get_cache_ttl("/hive:status", "medium")
        help_ttl = _get_cache_ttl("/hive:help", "medium")
        notifications_ttl = _get_cache_ttl("/hive:notifications", "medium")
        
        assert help_ttl > status_ttl, "Help content should cache longer than status"
        assert notifications_ttl < status_ttl, "Notifications should have shorter TTL"
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test performance metrics are collected correctly."""
        request = HiveCommandRequest(
            command="/hive:status --mobile",
            mobile_optimized=True,
            priority="high"
        )
        
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
            mock_agents.return_value = {}
            
            response = await execute_command(request)
            
            assert response.performance_metrics is not None
            assert "mobile_optimized" in response.performance_metrics
            assert "execution_time_ms" in response.performance_metrics
            assert "cache_eligible" in response.performance_metrics
            assert "priority" in response.performance_metrics
            
            # Mobile-specific metrics
            if response.mobile_optimized:
                assert "mobile_performance_score" in response.performance_metrics
                assert "mobile_response_time_target" in response.performance_metrics


class TestMobileOptimizationFeatures:
    """Test mobile-specific optimization features."""
    
    @pytest.mark.asyncio
    async def test_mobile_status_optimization(self):
        """Test mobile-optimized status response format."""
        status_command = HiveStatusCommand()
        
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
            mock_agents.return_value = {"agent1": {"role": "backend", "status": "active"}}
            
            # Test mobile-optimized response
            mobile_result = await status_command.execute(["--mobile"])
            
            assert mobile_result["success"]
            assert mobile_result["mobile_optimized"]
            assert "system_state" in mobile_result
            assert "quick_actions" in mobile_result
            assert "requires_attention" in mobile_result
            
            # Validate quick actions format
            quick_actions = mobile_result.get("quick_actions", [])
            for action in quick_actions:
                assert "action" in action
                assert "command" in action
                assert "description" in action
    
    @pytest.mark.asyncio
    async def test_intelligent_alert_filtering(self):
        """Test intelligent alert filtering for mobile relevance."""
        status_command = HiveStatusCommand()
        
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
            mock_agents.return_value = {}  # No agents to trigger alerts
            
            result = await status_command.execute(["--alerts-only", "--priority=high"])
            
            assert result["success"]
            assert "alerts" in result
            assert "alert_summary" in result
            
            # Test alert filtering
            alerts = result["alerts"]
            for alert in alerts:
                assert alert["priority"] in ["critical", "high"]
                assert "mobile_relevance_score" in alert
                assert "mobile_action" in alert
                assert "estimated_time" in alert
    
    @pytest.mark.asyncio
    async def test_context_aware_recommendations(self):
        """Test context-aware mobile recommendations."""
        focus_command = HiveFocusCommand()
        
        with patch('app.core.hive_slash_commands.HiveStatusCommand.execute') as mock_status:
            mock_status.return_value = {
                "alerts": [
                    {
                        "priority": "high",
                        "type": "insufficient_agents", 
                        "message": "Only 1 agent active",
                        "action": "Scale team",
                        "command": "/hive:start --team-size=5"
                    }
                ],
                "requires_action": True,
                "alert_summary": {"high": 1, "critical": 0}
            }
            
            mobile_result = await focus_command.execute(["development", "--mobile"])
            
            assert mobile_result["success"]
            assert mobile_result["mobile_optimized"]
            assert "quick_actions" in mobile_result
            assert "summary" in mobile_result
            
            # Validate mobile-optimized recommendations
            quick_actions = mobile_result["quick_actions"]
            assert len(quick_actions) <= 3, "Mobile should limit to 3 quick actions"
            
            for action in quick_actions:
                assert "time" in action, "Mobile actions should include time estimates"
                assert "priority" in action, "Mobile actions should include priority"
    
    @pytest.mark.asyncio 
    async def test_mobile_productivity_optimization(self):
        """Test mobile-optimized productivity insights."""
        productivity_command = HiveProductivityCommand()
        
        with patch('app.core.hive_slash_commands.HiveStatusCommand.execute') as mock_status:
            mock_status.return_value = {
                "agent_count": 3,
                "system_health": "healthy",
                "spawner_agents_detail": {"agent1": {"role": "backend"}},
                "orchestrator_agents_detail": {}
            }
            
            mobile_result = await productivity_command.execute(["--mobile", "--developer"])
            
            assert mobile_result["success"]
            assert mobile_result["mobile_optimized"]
            assert "productivity_summary" in mobile_result
            assert "quick_actions" in mobile_result
            
            # Validate mobile productivity format
            productivity_summary = mobile_result["productivity_summary"]
            assert "score" in productivity_summary
            assert "rating" in productivity_summary
            assert "status" in productivity_summary
            
            # Quick actions should be limited for mobile
            assert len(mobile_result["quick_actions"]) <= 3


class TestAgentCoordinationValidation:
    """Test agent coordination and multi-agent task routing."""
    
    @pytest.mark.asyncio
    async def test_hive_start_command_agent_spawning(self):
        """Test /hive:start command spawns appropriate agents."""
        registry = HiveSlashCommandRegistry()
        
        with patch('app.core.hive_slash_commands.spawn_development_team') as mock_spawn:
            with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                mock_spawn.return_value = {
                    "team_size": 5,
                    "roles": ["product_manager", "backend_developer", "frontend_developer", "qa_engineer", "devops_engineer"]
                }
                mock_agents.return_value = {
                    "agent1": {"role": "product_manager", "status": "active"},
                    "agent2": {"role": "backend_developer", "status": "active"},
                    "agent3": {"role": "frontend_developer", "status": "active"}
                }
                
                result = await registry.execute_command("/hive:start --team-size=5")
                
                assert result["success"]
                assert result["ready_for_development"]
                assert "team_composition" in result
                assert "active_agents" in result
                assert len(result["active_agents"]) >= 3
    
    @pytest.mark.asyncio
    async def test_agent_capability_matching(self):
        """Test agent spawning with specific capabilities."""
        registry = HiveSlashCommandRegistry()
        
        with patch('app.core.hive_slash_commands.get_agent_manager') as mock_manager:
            mock_agent_manager = AsyncMock()
            mock_agent_manager.spawn_agent.return_value = "agent_123"
            mock_manager.return_value = mock_agent_manager
            
            result = await registry.execute_command("/hive:spawn backend_developer --capabilities=api_development,database_design")
            
            assert result["success"]
            assert result["role"] == "backend_developer"
            assert "api_development" in result["capabilities"]
            assert "database_design" in result["capabilities"]
            assert result["agent_id"] == "agent_123"
    
    @pytest.mark.asyncio
    async def test_multi_agent_development_coordination(self):
        """Test autonomous development with multi-agent coordination."""
        registry = HiveSlashCommandRegistry()
        
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
            with patch('app.core.hive_slash_commands.spawn_development_team') as mock_spawn:
                with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                    # Setup mocks
                    mock_agents.return_value = {
                        "agent1": {"role": "product_manager"},
                        "agent2": {"role": "backend_developer"},
                        "agent3": {"role": "qa_engineer"}
                    }
                    mock_spawn.return_value = {"team_size": 3}
                    
                    # Mock successful subprocess execution
                    mock_process = AsyncMock()
                    mock_process.returncode = 0
                    mock_process.communicate.return_value = (b"Development completed", b"")
                    mock_subprocess.return_value = mock_process
                    
                    result = await registry.execute_command('/hive:develop "Build authentication API" --dashboard')
                    
                    assert result["success"]
                    assert result["project_description"] == "Build authentication API"
                    assert result["agents_involved"] == 3
                    assert result["dashboard_opened"]
    
    @pytest.mark.asyncio
    async def test_intelligent_task_routing(self):
        """Test intelligent task routing based on agent capabilities."""
        focus_command = HiveFocusCommand()
        
        # Test agent-specific task routing
        recommendations = await focus_command._generate_contextual_recommendations(
            status={"alerts": [], "requires_action": False},
            focus_area="development",
            priority_filter=None,
            target_agent="backend_developer",
            task_description="Implement user authentication"
        )
        
        assert len(recommendations) > 0
        agent_rec = recommendations[0]
        assert agent_rec["category"] == "agent_coordination"
        assert agent_rec["agent_id"] == "backend_developer"
        assert "authentication" in agent_rec["task_context"]
        assert "command" in agent_rec


class TestSystemIntegrationValidation:
    """Test system integration and real-time coordination."""
    
    @pytest.mark.asyncio
    async def test_hybrid_orchestrator_integration(self):
        """Test hybrid integration with both spawner and orchestrator agents."""
        status_command = HiveStatusCommand()
        
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_spawner:
            with patch('app.main.app') as mock_app:
                # Setup hybrid agent scenario
                mock_spawner.return_value = {
                    "spawner_agent1": {"role": "backend_developer", "status": "active"}
                }
                
                # Mock orchestrator integration
                mock_orchestrator = AsyncMock()
                mock_orchestrator.get_system_status.return_value = {
                    "orchestrator_agents": 2,
                    "agents": {
                        "orch_agent1": {"role": "product_manager"},
                        "orch_agent2": {"role": "qa_engineer"}
                    }
                }
                mock_app.state.orchestrator = mock_orchestrator
                
                result = await status_command.execute(["--detailed"])
                
                assert result["success"]
                assert result["hybrid_integration"]
                assert result["spawner_agents"] == 1
                assert result["orchestrator_agents"] == 2
                assert result["agent_count"] == 3
                assert result["system_ready"]  # 3 >= minimum of 3
    
    @pytest.mark.asyncio
    async def test_real_time_health_monitoring(self):
        """Test real-time system health monitoring integration."""
        status_command = HiveStatusCommand()
        
        with patch('requests.get') as mock_request:
            # Mock healthy system response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "components": {
                    "database": {"status": "healthy"},
                    "redis": {"status": "healthy"},
                    "agents": {"status": "healthy"}
                }
            }
            mock_response.elapsed.total_seconds.return_value = 0.045  # 45ms
            mock_request.return_value = mock_response
            
            result = await status_command.execute([])
            
            assert result["success"]
            assert result["system_health"] == "healthy"
            assert "components" in result
            assert result["response_time_ms"] == 45.0
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self):
        """Test performance degradation detection and alerting."""
        status_command = HiveStatusCommand()
        
        with patch('requests.get') as mock_request:
            with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                # Mock slow response scenario
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.elapsed.total_seconds.return_value = 0.150  # 150ms - slow
                mock_request.return_value = mock_response
                mock_agents.return_value = {"agent1": {"role": "backend"}}
                
                # Mock time to simulate slow response
                with patch('time.time') as mock_time:
                    mock_time.side_effect = [0.0, 0.120]  # 120ms response time
                    
                    result = await status_command.execute(["--alerts-only"])
                    
                    alerts = result.get("alerts", [])
                    performance_alerts = [a for a in alerts if a["type"] == "performance_degradation"]
                    
                    assert len(performance_alerts) > 0
                    perf_alert = performance_alerts[0]
                    assert perf_alert["priority"] == "medium"
                    assert "120ms" in perf_alert["message"]
                    assert perf_alert["mobile_action"] == "optimize_performance"


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    @pytest.mark.asyncio
    async def test_command_execution_error_handling(self):
        """Test graceful error handling in command execution."""
        registry = HiveSlashCommandRegistry()
        
        # Test invalid command
        result = await registry.execute_command("/hive:invalid_command")
        assert not result["success"]
        assert "Unknown command" in result["error"]
        assert "available_commands" in result
        
        # Test malformed command
        result = await registry.execute_command("/invalid:format")
        assert not result["success"]
        assert "Invalid hive command format" in result["error"]
    
    @pytest.mark.asyncio
    async def test_agent_spawn_failure_recovery(self):
        """Test recovery from agent spawning failures."""
        registry = HiveSlashCommandRegistry()
        
        with patch('app.core.hive_slash_commands.get_agent_manager') as mock_manager:
            mock_manager.side_effect = Exception("Agent manager unavailable")
            
            result = await registry.execute_command("/hive:spawn backend_developer")
            
            assert not result["success"]
            assert "Agent manager unavailable" in result["error"]
            assert "Failed to spawn backend_developer agent" in result["message"]
    
    @pytest.mark.asyncio
    async def test_system_health_check_failure_handling(self):
        """Test handling of system health check failures."""
        status_command = HiveStatusCommand()
        
        with patch('requests.get') as mock_request:
            # Mock connection failure
            mock_request.side_effect = Exception("Connection refused")
            
            result = await status_command.execute([])
            
            # Should still succeed but indicate health unknown
            assert result["success"]
            assert result["system_health"] == "unknown"
            assert "health_error" in result
    
    @pytest.mark.asyncio
    async def test_cache_failure_graceful_degradation(self):
        """Test graceful degradation when cache is unavailable."""
        request = HiveCommandRequest(
            command="/hive:status --mobile",
            mobile_optimized=True,
            use_cache=True
        )
        
        with patch('app.api.hive_commands.get_cached_mobile_response') as mock_cache:
            with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                # Mock cache failure
                mock_cache.side_effect = Exception("Cache unavailable")
                mock_agents.return_value = {"agent1": {"role": "backend"}}
                
                response = await execute_command(request)
                
                # Should fall back to live execution
                assert response.success
                assert not response.cached
                assert response.mobile_optimized
                assert response.execution_time_ms is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
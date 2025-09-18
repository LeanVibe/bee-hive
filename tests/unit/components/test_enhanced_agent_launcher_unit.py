"""
Unit Tests for EnhancedAgentLauncher - Component Isolation

Tests the EnhancedAgentLauncher component in complete isolation with all external
dependencies mocked. This ensures we test only the agent launcher's business logic
without any external system dependencies.

Testing Focus:
- Agent launch configuration and validation
- Different agent types (Claude Code, Cursor, etc.)
- Tmux session management integration
- Redis stream setup for agent communication
- Workspace and environment setup
- Error handling and recovery
- Resource cleanup

All external dependencies are mocked:
- Tmux session manager
- Redis streams manager
- File system operations
- Process execution
- Git operations
"""

import pytest
import asyncio
import uuid
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, Optional

# Component under test
from app.core.enhanced_agent_launcher import (
    EnhancedAgentLauncher,
    AgentLauncherType,
    AgentLaunchConfig,
    AgentLaunchResult,
    create_enhanced_agent_launcher
)

# Dependencies for mocking
from app.core.tmux_session_manager import SessionInfo, SessionStatus
from app.core.enhanced_redis_streams_manager import ConsumerGroupType
from app.models.agent import AgentStatus, AgentType


class TestEnhancedAgentLauncherUnit:
    """Unit tests for EnhancedAgentLauncher component in isolation."""

    @pytest.fixture
    def mock_tmux_manager(self):
        """Mock tmux session manager."""
        mock_tmux = AsyncMock()
        
        # Mock session creation
        mock_session_info = SessionInfo(
            session_id="test-session-123",
            session_name="test-agent-session",
            status=SessionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            windows=["main"],
            working_directory="/test/workspace"
        )
        
        mock_tmux.create_session.return_value = mock_session_info
        mock_tmux.terminate_session.return_value = True
        mock_tmux.get_session_info.return_value = mock_session_info
        mock_tmux.execute_command.return_value = {"exit_code": 0, "output": "success"}
        mock_tmux.get_session_metrics.return_value = {"active_sessions": 1}
        
        return mock_tmux

    @pytest.fixture
    def mock_redis_manager(self):
        """Mock enhanced Redis streams manager."""
        mock_redis = AsyncMock()
        
        mock_redis.create_consumer_group.return_value = True
        mock_redis.add_consumer_to_group.return_value = True
        mock_redis.send_message.return_value = "test-message-id"
        mock_redis.get_stream_info.return_value = {"consumers": 1}
        
        return mock_redis

    @pytest.fixture
    def mock_short_id_generator(self):
        """Mock short ID generator."""
        mock_generator = Mock()
        mock_generator.generate.return_value = "TEST123"
        return mock_generator

    @pytest.fixture
    def isolated_agent_launcher(
        self,
        mock_tmux_manager,
        mock_redis_manager,
        mock_short_id_generator
    ):
        """Create EnhancedAgentLauncher with all dependencies mocked."""
        return EnhancedAgentLauncher(
            tmux_manager=mock_tmux_manager,
            redis_manager=mock_redis_manager,
            short_id_generator=mock_short_id_generator
        )

    @pytest.fixture
    def claude_code_config(self):
        """Standard Claude Code launch configuration."""
        return AgentLaunchConfig(
            agent_type=AgentLauncherType.CLAUDE_CODE,
            task_id="test-task-123",
            workspace_name="test-workspace",
            git_branch="feature/test-branch",
            working_directory="/test/project",
            environment_vars={"TEST_VAR": "test_value"}
        )

    class TestInitialization:
        """Test agent launcher initialization."""

        def test_launcher_creation(self, isolated_agent_launcher):
            """Test that launcher creates correctly with mocked dependencies."""
            assert isolated_agent_launcher.tmux_manager is not None
            assert isolated_agent_launcher.redis_manager is not None
            assert isolated_agent_launcher.short_id_generator is not None
            assert hasattr(isolated_agent_launcher, '_active_agents')

        def test_launcher_attributes(self, isolated_agent_launcher):
            """Test launcher has expected attributes."""
            assert hasattr(isolated_agent_launcher, '_launch_metrics')
            assert hasattr(isolated_agent_launcher, '_recovery_attempts')
            assert isolated_agent_launcher._active_agents == {}

    class TestAgentLaunchConfig:
        """Test agent launch configuration validation."""

        def test_config_creation_claude_code(self):
            """Test creating Claude Code configuration."""
            config = AgentLaunchConfig(
                agent_type=AgentLauncherType.CLAUDE_CODE,
                task_id="test-task"
            )
            
            assert config.agent_type == AgentLauncherType.CLAUDE_CODE
            assert config.task_id == "test-task"
            assert config.consumer_group == ConsumerGroupType.GENERAL_AGENTS

        def test_config_creation_cursor_agent(self):
            """Test creating Cursor agent configuration."""
            config = AgentLaunchConfig(
                agent_type=AgentLauncherType.CURSOR_AGENT,
                workspace_name="cursor-workspace",
                git_branch="main"
            )
            
            assert config.agent_type == AgentLauncherType.CURSOR_AGENT
            assert config.workspace_name == "cursor-workspace"
            assert config.git_branch == "main"

        def test_config_to_dict_serialization(self):
            """Test configuration serialization to dictionary."""
            config = AgentLaunchConfig(
                agent_type=AgentLauncherType.AIDER,
                task_id="aider-task",
                environment_vars={"API_KEY": "secret"}
            )
            
            config_dict = config.to_dict()
            assert config_dict["agent_type"] == "aider"
            assert config_dict["task_id"] == "aider-task"
            assert config_dict["environment_vars"]["API_KEY"] == "secret"
            assert config_dict["consumer_group"] == ConsumerGroupType.GENERAL_AGENTS.value

        def test_config_with_custom_agent_config(self):
            """Test configuration with custom agent-specific settings."""
            custom_config = {
                "model": "claude-3-opus",
                "temperature": 0.7,
                "max_tokens": 4000
            }
            
            config = AgentLaunchConfig(
                agent_type=AgentLauncherType.CUSTOM,
                agent_config=custom_config
            )
            
            assert config.agent_config == custom_config
            assert config.agent_config["model"] == "claude-3-opus"

    class TestAgentLaunchProcess:
        """Test the agent launching process."""

        @pytest.mark.asyncio
        async def test_launch_agent_claude_code_success(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test successful Claude Code agent launch."""
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-claude-agent"
                )
                
                assert result.success is True
                assert result.agent_id is not None
                assert result.session_id == "test-session-123"
                assert result.session_name == "test-agent-session"
                assert result.workspace_path is not None
                assert result.error_message is None
                
                # Verify tmux session was created
                mock_tmux_manager.create_session.assert_called_once()

        @pytest.mark.asyncio
        async def test_launch_agent_cursor_agent_success(
            self,
            isolated_agent_launcher,
            mock_tmux_manager
        ):
            """Test successful Cursor agent launch."""
            cursor_config = AgentLaunchConfig(
                agent_type=AgentLauncherType.CURSOR_AGENT,
                workspace_name="cursor-workspace"
            )
            
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=cursor_config,
                    agent_name="test-cursor-agent"
                )
                
                assert result.success is True
                assert result.agent_id is not None
                mock_tmux_manager.create_session.assert_called_once()

        @pytest.mark.asyncio
        async def test_launch_agent_aider_success(
            self,
            isolated_agent_launcher,
            mock_tmux_manager
        ):
            """Test successful Aider agent launch."""
            aider_config = AgentLaunchConfig(
                agent_type=AgentLauncherType.AIDER,
                workspace_name="aider-workspace",
                environment_vars={"OPENAI_API_KEY": "test-key"}
            )
            
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=aider_config,
                    agent_name="test-aider-agent"
                )
                
                assert result.success is True
                mock_tmux_manager.create_session.assert_called_once()

        @pytest.mark.asyncio
        async def test_launch_agent_workspace_setup(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test that workspace is properly set up during launch."""
            with patch('app.core.enhanced_agent_launcher.Path.mkdir') as mock_mkdir, \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=False), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-workspace-agent"
                )
                
                assert result.success is True
                # Verify workspace directory was created
                mock_mkdir.assert_called()

        @pytest.mark.asyncio
        async def test_launch_agent_git_branch_setup(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test that git branch is properly set up during launch."""
            claude_code_config.git_branch = "feature/new-feature"
            
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-git-agent"
                )
                
                assert result.success is True
                # Verify git operations were called
                assert any("git" in str(call) for call in mock_subprocess.call_args_list)

        @pytest.mark.asyncio
        async def test_launch_agent_redis_stream_setup(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_redis_manager
        ):
            """Test that Redis streams are properly configured during launch."""
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-redis-agent"
                )
                
                assert result.success is True
                # Verify Redis stream operations
                mock_redis_manager.create_consumer_group.assert_called()
                mock_redis_manager.add_consumer_to_group.assert_called()

    class TestAgentLaunchFailures:
        """Test agent launch failure scenarios."""

        @pytest.mark.asyncio
        async def test_launch_agent_tmux_failure(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test handling of tmux session creation failure."""
            # Mock tmux failure
            mock_tmux_manager.create_session.side_effect = Exception("Tmux creation failed")
            
            result = await isolated_agent_launcher.launch_agent(
                config=claude_code_config,
                agent_name="test-tmux-fail-agent"
            )
            
            assert result.success is False
            assert "Tmux creation failed" in result.error_message

        @pytest.mark.asyncio
        async def test_launch_agent_workspace_creation_failure(
            self,
            isolated_agent_launcher,
            claude_code_config
        ):
            """Test handling of workspace creation failure."""
            with patch('app.core.enhanced_agent_launcher.Path.mkdir', side_effect=OSError("Permission denied")):
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-workspace-fail-agent"
                )
                
                assert result.success is False
                assert "Permission denied" in result.error_message

        @pytest.mark.asyncio
        async def test_launch_agent_git_failure(
            self,
            isolated_agent_launcher,
            claude_code_config
        ):
            """Test handling of git operation failure."""
            claude_code_config.git_branch = "invalid/branch"
            
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                # Mock git failure
                mock_subprocess.return_value = Mock(returncode=1, stdout="", stderr="Git error")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-git-fail-agent"
                )
                
                assert result.success is False
                assert "Git error" in result.error_message

        @pytest.mark.asyncio
        async def test_launch_agent_redis_failure(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_redis_manager
        ):
            """Test handling of Redis setup failure."""
            # Mock Redis failure
            mock_redis_manager.create_consumer_group.side_effect = Exception("Redis connection failed")
            
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-redis-fail-agent"
                )
                
                # Should still succeed with Redis failure (graceful degradation)
                assert result.success is True

    class TestAgentTermination:
        """Test agent termination and cleanup."""

        @pytest.mark.asyncio
        async def test_terminate_agent_success(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test successful agent termination."""
            # First launch an agent
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-terminate-agent"
                )
                
                agent_id = result.agent_id
                
                # Now terminate it
                termination_result = await isolated_agent_launcher.terminate_agent(
                    agent_id=agent_id,
                    cleanup_workspace=True
                )
                
                assert termination_result is True
                mock_tmux_manager.terminate_session.assert_called()

        @pytest.mark.asyncio
        async def test_terminate_nonexistent_agent(
            self,
            isolated_agent_launcher
        ):
            """Test terminating a non-existent agent."""
            result = await isolated_agent_launcher.terminate_agent(
                agent_id="non-existent-id",
                cleanup_workspace=False
            )
            
            assert result is False

        @pytest.mark.asyncio
        async def test_terminate_agent_with_workspace_cleanup(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test agent termination with workspace cleanup."""
            # Launch agent first
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess, \
                 patch('shutil.rmtree') as mock_rmtree:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-cleanup-agent"
                )
                
                # Terminate with cleanup
                await isolated_agent_launcher.terminate_agent(
                    agent_id=result.agent_id,
                    cleanup_workspace=True
                )
                
                # Verify workspace cleanup was attempted
                mock_rmtree.assert_called()

    class TestAgentStatus:
        """Test agent status monitoring."""

        @pytest.mark.asyncio
        async def test_get_agent_status_active(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test getting status of active agent."""
            # Launch agent first
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-status-agent"
                )
                
                # Get agent status
                status = await isolated_agent_launcher.get_agent_status(result.agent_id)
                
                assert status is not None
                assert "status" in status
                assert "session_info" in status

        @pytest.mark.asyncio
        async def test_get_agent_status_nonexistent(
            self,
            isolated_agent_launcher
        ):
            """Test getting status of non-existent agent."""
            status = await isolated_agent_launcher.get_agent_status("non-existent-id")
            assert status is None

        @pytest.mark.asyncio
        async def test_get_launcher_metrics(
            self,
            isolated_agent_launcher
        ):
            """Test getting launcher performance metrics."""
            metrics = await isolated_agent_launcher.get_launcher_metrics()
            
            assert isinstance(metrics, dict)
            assert "launched" in metrics
            assert "terminated" in metrics
            assert "active" in metrics
            assert "failed_launches" in metrics

    class TestPerformanceTracking:
        """Test performance tracking and metrics."""

        @pytest.mark.asyncio
        async def test_launch_performance_tracking(
            self,
            isolated_agent_launcher,
            claude_code_config
        ):
            """Test that launch performance is tracked."""
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-perf-agent"
                )
                
                assert result.launch_time_seconds >= 0
                assert isinstance(result.launch_time_seconds, float)

        @pytest.mark.asyncio
        async def test_multiple_launches_metrics(
            self,
            isolated_agent_launcher,
            claude_code_config
        ):
            """Test metrics tracking across multiple launches."""
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                # Launch multiple agents
                for i in range(3):
                    await isolated_agent_launcher.launch_agent(
                        config=claude_code_config,
                        agent_name=f"test-multi-agent-{i}"
                    )
                
                metrics = await isolated_agent_launcher.get_launcher_metrics()
                assert metrics["launched"] == 3
                assert metrics["active"] == 3

    class TestErrorRecovery:
        """Test error recovery and resilience."""

        @pytest.mark.asyncio
        async def test_recovery_from_partial_failure(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager,
            mock_redis_manager
        ):
            """Test recovery from partial launch failure."""
            # Mock Redis failure but tmux success
            mock_redis_manager.create_consumer_group.side_effect = Exception("Redis failed")
            
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run') as mock_subprocess:
                
                mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-recovery-agent"
                )
                
                # Should still succeed with partial failure
                assert result.success is True
                # But should track the Redis issue
                metrics = await isolated_agent_launcher.get_launcher_metrics()
                # Could track warnings or partial failures in metrics

        @pytest.mark.asyncio
        async def test_cleanup_on_launch_failure(
            self,
            isolated_agent_launcher,
            claude_code_config,
            mock_tmux_manager
        ):
            """Test that resources are cleaned up on launch failure."""
            # Mock failure after session creation
            mock_tmux_manager.create_session.return_value = SessionInfo(
                session_id="temp-session",
                session_name="temp-session-name",
                status=SessionStatus.ACTIVE,
                created_at=datetime.utcnow(),
                windows=["main"],
                working_directory="/temp/workspace"
            )
            
            with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
                 patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
                 patch('subprocess.run', side_effect=Exception("Process launch failed")):
                
                result = await isolated_agent_launcher.launch_agent(
                    config=claude_code_config,
                    agent_name="test-cleanup-fail-agent"
                )
                
                assert result.success is False
                # Should attempt cleanup of created session
                mock_tmux_manager.terminate_session.assert_called()


class TestFactoryFunction:
    """Test factory function for agent launcher creation."""

    @pytest.mark.asyncio
    async def test_create_enhanced_agent_launcher(self):
        """Test factory function for creating enhanced agent launcher."""
        with patch('app.core.enhanced_agent_launcher.TmuxSessionManager') as mock_tmux, \
             patch('app.core.enhanced_agent_launcher.EnhancedRedisStreamsManager') as mock_redis, \
             patch('app.core.enhanced_agent_launcher.ShortIdGenerator') as mock_id_gen:
            
            mock_tmux_instance = AsyncMock()
            mock_redis_instance = AsyncMock() 
            mock_id_gen_instance = Mock()
            
            mock_tmux.return_value = mock_tmux_instance
            mock_redis.return_value = mock_redis_instance
            mock_id_gen.return_value = mock_id_gen_instance
            
            launcher = await create_enhanced_agent_launcher(
                tmux_manager=mock_tmux_instance,
                redis_manager=mock_redis_instance,
                short_id_generator=mock_id_gen_instance
            )
            
            assert isinstance(launcher, EnhancedAgentLauncher)
            assert launcher.tmux_manager == mock_tmux_instance
            assert launcher.redis_manager == mock_redis_instance
            assert launcher.short_id_generator == mock_id_gen_instance


class TestAgentTypeHandling:
    """Test handling of different agent types."""

    @pytest.fixture
    def isolated_launcher_for_types(self, mock_tmux_manager, mock_redis_manager, mock_short_id_generator):
        """Launcher specifically for testing different agent types."""
        return EnhancedAgentLauncher(
            tmux_manager=mock_tmux_manager,
            redis_manager=mock_redis_manager,
            short_id_generator=mock_short_id_generator
        )

    @pytest.mark.parametrize("agent_type", [
        AgentLauncherType.CLAUDE_CODE,
        AgentLauncherType.CURSOR_AGENT,
        AgentLauncherType.AIDER,
        AgentLauncherType.CONTINUE,
        AgentLauncherType.CUSTOM
    ])
    @pytest.mark.asyncio
    async def test_launch_different_agent_types(
        self,
        isolated_launcher_for_types,
        agent_type
    ):
        """Test launching different types of agents."""
        config = AgentLaunchConfig(
            agent_type=agent_type,
            workspace_name=f"{agent_type.value}-workspace"
        )
        
        with patch('app.core.enhanced_agent_launcher.Path.mkdir'), \
             patch('app.core.enhanced_agent_launcher.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_subprocess:
            
            mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
            
            result = await isolated_launcher_for_types.launch_agent(
                config=config,
                agent_name=f"test-{agent_type.value}-agent"
            )
            
            assert result.success is True
            assert result.agent_id is not None
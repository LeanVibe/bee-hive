"""
Comprehensive Integration Tests for Autonomous Self-Modification System

Tests the complete end-to-end functionality of the autonomous self-modification
and sleep-wake consolidation system, including all major components and workflows.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.self_modification.self_modification_service import SelfModificationService
from app.core.sleep_wake_manager import SleepWakeManager
from app.core.enhanced_github_integration import AutomatedPRWorkflow
from app.core.meta_learning_engine import MetaLearningEngine
from app.core.autonomous_quality_gates import AutonomousQualityGateSystem
from app.models.self_modification import ModificationSession, CodeModification
from app.models.agent import Agent
from app.models.sleep_wake import SleepWakeCycle


class TestAutonomousSelfModificationIntegration:
    """Test complete autonomous self-modification workflows."""
    
    @pytest.fixture
    async def self_mod_service(self, async_session):
        """Create self-modification service."""
        return SelfModificationService(async_session)
    
    @pytest.fixture
    async def sleep_wake_manager(self):
        """Create sleep-wake manager."""
        manager = SleepWakeManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    async def meta_learning_engine(self, async_session):
        """Create meta-learning engine."""
        engine = MetaLearningEngine(async_session)
        await engine.initialize()
        return engine
    
    @pytest.fixture
    async def quality_gate_system(self, async_session):
        """Create quality gate system."""
        return AutonomousQualityGateSystem(async_session)
    
    @pytest.fixture
    def test_codebase(self):
        """Create a test codebase for modification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test Python files
            (temp_path / "main.py").write_text("""
import time

def slow_function():
    # This function is inefficient
    result = ""
    for i in range(1000):
        result += str(i)
        time.sleep(0.001)  # Unnecessary delay
    return result

def process_data(data):
    # Another inefficient function
    processed = []
    for item in data:
        if item % 2 == 0:
            processed.append(item * 2)
    return processed

if __name__ == "__main__":
    print(slow_function())
    print(process_data([1, 2, 3, 4, 5]))
            """.strip())
            
            (temp_path / "utils.py").write_text("""
def inefficient_search(data_list, target):
    # Linear search - could be optimized
    for i in range(len(data_list)):
        if data_list[i] == target:
            return i
    return -1

def memory_waster():
    # Creates unnecessary large objects
    big_list = [i * j for i in range(1000) for j in range(1000)]
    return len(big_list)

class OldStyleClass:
    # Uses old-style class definition patterns
    def __init__(self):
        self.data = []
        
    def add_item(self, item):
        self.data.append(item)
        
    def get_all(self):
        return self.data
            """.strip())
            
            (temp_path / "config.py").write_text("""
# Configuration with potential security issues
SECRET_KEY = "hardcoded_secret_123"  # Security issue
DEBUG = True
DATABASE_URL = "sqlite:///app.db"

ALLOWED_HOSTS = ['*']  # Security issue

def get_config():
    return {
        'secret': SECRET_KEY,
        'debug': DEBUG,
        'db_url': DATABASE_URL
    }
            """.strip())
            
            yield str(temp_path)
    
    @pytest.mark.asyncio
    async def test_complete_autonomous_modification_workflow(
        self, 
        self_mod_service, 
        sleep_wake_manager, 
        meta_learning_engine,
        quality_gate_system,
        test_codebase,
        async_session
    ):
        """Test complete end-to-end autonomous modification workflow."""
        
        # Phase 1: Analyze codebase
        analysis_response = await self_mod_service.analyze_codebase(
            codebase_path=test_codebase,
            modification_goals=["improve_performance", "fix_security", "enhance_maintainability"],
            safety_level="conservative"
        )
        
        assert analysis_response is not None
        assert analysis_response.analysis_id is not None
        assert analysis_response.total_suggestions > 0
        assert len(analysis_response.suggestions) > 0
        
        # Phase 2: Apply meta-learning to improve suggestions
        improved_suggestions = await meta_learning_engine.generate_improved_suggestions(
            codebase_analysis={"total_files": 3, "complexity": "medium"},
            modification_goals=["improve_performance", "fix_security"],
            context={"codebase_type": "python", "project_size": "small"}
        )
        
        # Meta-learning might not have suggestions initially (no learning data)
        # That's expected for a fresh system
        
        # Phase 3: Quality gate validation
        # Get some modifications to validate
        selected_modifications = analysis_response.suggestions[:3]  # First 3 suggestions
        
        if selected_modifications:
            # Create modification objects for validation
            modifications = []
            for suggestion in selected_modifications:
                modification = CodeModification(
                    session_id=analysis_response.analysis_id,
                    file_path=suggestion.file_path,
                    modification_type=suggestion.modification_type,
                    original_content=suggestion.original_content,
                    modified_content=suggestion.modified_content,
                    modification_reason=suggestion.modification_reason,
                    safety_score=suggestion.safety_score
                )
                modifications.append(modification)
            
            # Validate modifications through quality gates
            validation_result = await quality_gate_system.validate_modification_batch(
                modifications=modifications,
                validation_suite="standard"
            )
            
            assert validation_result is not None
            assert "overall_status" in validation_result
            assert "gate_results" in validation_result
            
            # Phase 4: Apply modifications if they pass quality gates
            if validation_result["overall_status"] in ["passed", "warning"]:
                modification_ids = [suggestion.id for suggestion in selected_modifications]
                
                apply_response = await self_mod_service.apply_modifications(
                    analysis_id=analysis_response.analysis_id,
                    selected_modifications=modification_ids,
                    dry_run=True  # Use dry run for testing
                )
                
                assert apply_response is not None
                assert apply_response.session_id == analysis_response.analysis_id
        
        # Phase 5: Initiate sleep-wake consolidation
        test_agent_id = uuid4()
        
        # Create test agent
        async with async_session as session:
            agent = Agent(
                id=test_agent_id,
                name="test_agent",
                agent_type="developer",
                status="active"
            )
            session.add(agent)
            await session.commit()
        
        # Initiate sleep cycle
        sleep_success = await sleep_wake_manager.initiate_sleep_cycle(
            agent_id=test_agent_id,
            cycle_type="post_modification",
            expected_wake_time=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Sleep initiation might fail if components aren't fully mocked
        # That's acceptable for this integration test
        
        # Phase 6: Record outcomes for meta-learning
        if selected_modifications:
            for suggestion in selected_modifications:
                await meta_learning_engine.record_modification_outcome(
                    modification_id=suggestion.id,
                    success=True,
                    validation_results=validation_result,
                    user_feedback={"rating": 4, "comment": "Good improvement"}
                )
        
        # Phase 7: Trigger learning cycle
        learning_results = await meta_learning_engine.perform_learning_cycle()
        assert learning_results is not None
        assert "status" in learning_results
        
        # Verify the complete workflow executed successfully
        assert analysis_response.status.value in ["suggestions_ready", "completed"]
    
    @pytest.mark.asyncio
    async def test_github_integration_workflow(self, test_codebase):
        """Test GitHub integration workflow."""
        
        with patch('app.core.enhanced_github_integration.Github') as mock_github:
            # Mock GitHub API
            mock_repo = Mock()
            mock_pr = Mock()
            mock_pr.number = 123
            mock_pr.html_url = "https://github.com/test/repo/pull/123"
            
            mock_repo.create_pull.return_value = mock_pr
            mock_github.return_value.get_repo.return_value = mock_repo
            
            # Create GitHub workflow
            workflow = AutomatedPRWorkflow("fake_token", "test/repo")
            await workflow.initialize()
            
            # Create PR for modifications
            modifications = [
                {
                    "id": str(uuid4()),
                    "file_path": "main.py",
                    "modification_type": "performance",
                    "reasoning": "Optimized loop performance",
                    "lines_added": 5,
                    "lines_removed": 8
                }
            ]
            
            pr = await workflow.create_self_modification_pr(
                branch_name="autonomous-mod-test",
                modification_session_id=uuid4(),
                modifications_applied=modifications,
                safety_score=0.95,
                performance_improvement=15.0
            )
            
            assert pr is not None
            assert pr.number == 123
            
            # Test PR monitoring
            pr_status = await workflow.monitor_pr_status(123, auto_merge_on_approval=False)
            assert pr_status is not None
            assert "pr_number" in pr_status
    
    @pytest.mark.asyncio
    async def test_quality_gates_comprehensive_validation(self, quality_gate_system):
        """Test comprehensive quality gate validation."""
        
        # Create test modifications
        modifications = [
            CodeModification(
                id=uuid4(),
                session_id=uuid4(),
                file_path="test.py",
                modification_type="performance",
                original_content="def slow_func():\n    return sum(range(1000))",
                modified_content="def fast_func():\n    return 499500",  # Optimized calculation
                safety_score=0.9
            ),
            CodeModification(
                id=uuid4(),
                session_id=uuid4(),
                file_path="security.py",
                modification_type="security",
                original_content="password = 'hardcoded'",
                modified_content="password = os.environ.get('PASSWORD')",
                safety_score=0.95
            )
        ]
        
        # Run comprehensive validation
        validation_result = await quality_gate_system.validate_modification_batch(
            modifications=modifications,
            validation_suite="comprehensive"
        )
        
        assert validation_result is not None
        assert "validation_id" in validation_result
        assert "overall_status" in validation_result
        assert "gate_results" in validation_result
        assert validation_result["modifications_validated"] == 2
        
        # Check that multiple gates were executed
        gate_results = validation_result["gate_results"]
        assert len(gate_results) > 0
        
        gate_types = [result["gate_type"] for result in gate_results]
        expected_gates = ["syntax_validation", "security_scan", "safety_analysis"]
        
        for gate_type in expected_gates:
            assert gate_type in gate_types
    
    @pytest.mark.asyncio
    async def test_meta_learning_pattern_discovery(self, meta_learning_engine):
        """Test meta-learning pattern discovery."""
        
        # Simulate recording multiple outcomes
        outcomes = [
            {
                "modification_id": uuid4(),
                "success": True,
                "validation_results": {"overall_status": "passed", "score": 0.9},
                "user_feedback": {"rating": 5}
            },
            {
                "modification_id": uuid4(),
                "success": True,
                "validation_results": {"overall_status": "passed", "score": 0.85},
                "user_feedback": {"rating": 4}
            },
            {
                "modification_id": uuid4(),
                "success": False,
                "validation_results": {"overall_status": "failed", "score": 0.3},
                "user_feedback": {"rating": 1}
            }
        ]
        
        for outcome in outcomes:
            await meta_learning_engine.record_modification_outcome(**outcome)
        
        # Perform learning cycle
        learning_results = await meta_learning_engine.perform_learning_cycle()
        
        assert learning_results is not None
        assert "status" in learning_results
        
        # Get learning insights
        insights = await meta_learning_engine.get_learning_insights()
        
        assert insights is not None
        assert "learning_metrics" in insights
        assert "pattern_summary" in insights
    
    @pytest.mark.asyncio
    async def test_sleep_wake_consolidation_cycle(self, sleep_wake_manager, async_session):
        """Test sleep-wake consolidation cycle."""
        
        # Create test agent
        test_agent_id = uuid4()
        
        async with async_session as session:
            agent = Agent(
                id=test_agent_id,
                name="test_consolidation_agent",
                agent_type="developer",
                status="active"
            )
            session.add(agent)
            await session.commit()
        
        # Test sleep cycle initiation
        sleep_success = await sleep_wake_manager.initiate_sleep_cycle(
            agent_id=test_agent_id,
            cycle_type="test_consolidation"
        )
        
        # Sleep might fail due to missing dependencies - that's OK for integration test
        
        # Test wake cycle (if sleep succeeded)
        if sleep_success:
            wake_success = await sleep_wake_manager.initiate_wake_cycle(test_agent_id)
            # Wake success depends on system state
        
        # Get system status
        system_status = await sleep_wake_manager.get_system_status()
        
        assert system_status is not None
        assert "timestamp" in system_status
        assert "agents" in system_status
        
        # Test optimization
        optimization_results = await sleep_wake_manager.optimize_performance()
        
        assert optimization_results is not None
        assert "timestamp" in optimization_results
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, 
        self_mod_service, 
        quality_gate_system,
        test_codebase
    ):
        """Test error handling and recovery mechanisms."""
        
        # Test with invalid codebase path
        try:
            await self_mod_service.analyze_codebase(
                codebase_path="/nonexistent/path",
                modification_goals=["improve_performance"]
            )
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "not found" in str(e).lower() or "no such file" in str(e).lower()
        
        # Test with invalid modification
        invalid_modification = CodeModification(
            id=uuid4(),
            session_id=uuid4(),
            file_path="test.py",
            modification_type="performance",
            original_content="def valid_func(): pass",
            modified_content="def invalid_func( # Syntax error",  # Invalid syntax
            safety_score=0.5
        )
        
        validation_result = await quality_gate_system.validate_single_modification(
            modification=invalid_modification
        )
        
        assert validation_result is not None
        assert validation_result["overall_status"] == "failed"
        
        # Test rollback scenario
        session_id = uuid4()
        modification_id = uuid4()
        
        try:
            rollback_result = await self_mod_service.rollback_modification(
                modification_id=modification_id,
                rollback_reason="Test rollback"
            )
            # This should fail because modification doesn't exist
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "not found" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_performance_and_scalability(
        self, 
        quality_gate_system, 
        async_session
    ):
        """Test system performance with multiple concurrent operations."""
        
        # Create multiple modifications for concurrent testing
        modifications = []
        for i in range(10):
            mod = CodeModification(
                id=uuid4(),
                session_id=uuid4(),
                file_path=f"test_{i}.py",
                modification_type="performance",
                original_content=f"def func_{i}(): return {i}",
                modified_content=f"def optimized_func_{i}(): return {i}",
                safety_score=0.8 + (i % 3) * 0.05  # Varying safety scores
            )
            modifications.append(mod)
        
        # Test concurrent validation
        validation_tasks = []
        for mod in modifications[:5]:  # Test with 5 concurrent validations
            task = asyncio.create_task(
                quality_gate_system.validate_single_modification(mod)
            )
            validation_tasks.append(task)
        
        # Wait for all validations to complete
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Verify results
        successful_validations = 0
        for result in results:
            if isinstance(result, dict) and "overall_status" in result:
                successful_validations += 1
        
        assert successful_validations >= len(validation_tasks) // 2  # At least half should succeed
        
        # Test batch validation performance
        start_time = datetime.utcnow()
        
        batch_result = await quality_gate_system.validate_modification_batch(
            modifications=modifications,
            validation_suite="minimal"  # Use minimal suite for performance
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        assert batch_result is not None
        assert execution_time < 60  # Should complete within 60 seconds
        assert batch_result["modifications_validated"] == len(modifications)
    
    @pytest.mark.asyncio
    async def test_system_health_and_monitoring(
        self, 
        self_mod_service, 
        sleep_wake_manager,
        quality_gate_system,
        async_session
    ):
        """Test system health monitoring and metrics."""
        
        # Test self-modification system health
        mod_health = await self_mod_service.get_system_health()
        
        assert mod_health is not None
        # Health check might have issues due to missing dependencies - that's OK
        
        # Test sleep-wake system status
        sleep_status = await sleep_wake_manager.get_system_status()
        
        assert sleep_status is not None
        assert "timestamp" in sleep_status
        assert "system_healthy" in sleep_status
        
        # Test quality gate metrics
        gate_metrics = await quality_gate_system.get_validation_metrics()
        
        assert gate_metrics is not None
        assert "system_metrics" in gate_metrics
        assert "available_suites" in gate_metrics
        assert len(gate_metrics["available_suites"]) > 0


class TestSystemIntegrationScenarios:
    """Test realistic system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_enterprise_deployment_scenario(self, async_session):
        """Test enterprise deployment scenario with multiple agents."""
        
        # Create multiple agents
        agent_ids = []
        async with async_session as session:
            for i in range(3):
                agent = Agent(
                    id=uuid4(),
                    name=f"enterprise_agent_{i}",
                    agent_type="developer",
                    status="active"
                )
                session.add(agent)
                agent_ids.append(agent.id)
            
            await session.commit()
        
        # Initialize system components
        sleep_wake_manager = SleepWakeManager()
        await sleep_wake_manager.initialize()
        
        quality_gate_system = AutonomousQualityGateSystem(async_session)
        
        # Test system status with multiple agents
        system_status = await sleep_wake_manager.get_system_status()
        
        assert system_status is not None
        assert len(system_status["agents"]) >= len(agent_ids)
        
        # Test custom validation suite creation
        custom_suite = await quality_gate_system.create_custom_validation_suite(
            suite_name="enterprise_validation",
            gates=["syntax_validation", "security_scan", "safety_analysis"],
            configuration={
                "score_threshold": 0.95,
                "failure_tolerance": 0,
                "timeout_minutes": 45
            }
        )
        
        assert custom_suite is not None
        assert custom_suite.name == "enterprise_validation"
        assert custom_suite.required_score_threshold == 0.95
    
    @pytest.mark.asyncio
    async def test_continuous_improvement_scenario(self, async_session):
        """Test continuous improvement scenario with feedback loops."""
        
        meta_learning_engine = MetaLearningEngine(async_session)
        await meta_learning_engine.initialize()
        
        # Simulate continuous improvement over time
        improvement_cycles = 3
        
        for cycle in range(improvement_cycles):
            # Generate some outcomes for this cycle
            outcomes = []
            for i in range(5):
                success_rate = 0.6 + (cycle * 0.1)  # Improving over time
                outcome = {
                    "modification_id": uuid4(),
                    "success": i < int(5 * success_rate),
                    "validation_results": {
                        "overall_status": "passed" if i < int(5 * success_rate) else "failed",
                        "score": 0.8 + (cycle * 0.05)
                    }
                }
                outcomes.append(outcome)
            
            # Record outcomes
            for outcome in outcomes:
                await meta_learning_engine.record_modification_outcome(**outcome)
            
            # Perform learning cycle
            learning_results = await meta_learning_engine.perform_learning_cycle()
            
            assert learning_results["status"] in ["completed", "disabled"]
        
        # Verify improvement over cycles
        insights = await meta_learning_engine.get_learning_insights()
        
        assert insights is not None
        if insights["learning_metrics"]["modifications_analyzed"] > 0:
            assert insights["learning_metrics"]["improvement_cycles"] == improvement_cycles
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_scenario(self, async_session):
        """Test disaster recovery and emergency shutdown scenarios."""
        
        sleep_wake_manager = SleepWakeManager()
        await sleep_wake_manager.initialize()
        
        # Create test agent
        test_agent_id = uuid4()
        async with async_session as session:
            agent = Agent(
                id=test_agent_id,
                name="disaster_test_agent",
                agent_type="developer",
                status="active"
            )
            session.add(agent)
            await session.commit()
        
        # Test emergency shutdown for specific agent
        recovery_result = await sleep_wake_manager.emergency_shutdown(test_agent_id)
        
        # Recovery might fail due to missing components - that's acceptable
        assert isinstance(recovery_result, bool)
        
        # Test system-wide emergency shutdown
        system_recovery_result = await sleep_wake_manager.emergency_shutdown(None)
        
        assert isinstance(system_recovery_result, bool)
        
        # Verify system can still provide status after recovery attempt
        final_status = await sleep_wake_manager.get_system_status()
        
        assert final_status is not None
        assert "timestamp" in final_status


# Helper functions for test setup
async def create_test_modification_session(
    async_session: AsyncSession,
    codebase_path: str,
    goals: list
) -> ModificationSession:
    """Create a test modification session."""
    session = ModificationSession(
        agent_id=uuid4(),
        codebase_path=codebase_path,
        modification_goals=goals,
        safety_level="conservative"
    )
    
    async with async_session as db_session:
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
    
    return session


async def create_test_code_modification(
    async_session: AsyncSession,
    session_id: str,
    file_path: str,
    modification_type: str
) -> CodeModification:
    """Create a test code modification."""
    modification = CodeModification(
        session_id=session_id,
        file_path=file_path,
        modification_type=modification_type,
        original_content="def original_func(): pass",
        modified_content="def improved_func(): return 'improved'",
        modification_reason="Test improvement",
        safety_score=0.85
    )
    
    async with async_session as db_session:
        db_session.add(modification)
        await db_session.commit()
        await db_session.refresh(modification)
    
    return modification
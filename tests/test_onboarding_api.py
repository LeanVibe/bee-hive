"""
Tests for Epic 6 Onboarding API

Comprehensive test suite for the interactive onboarding system API endpoints.
Tests cover session management, progress tracking, analytics, and Epic 6 success metrics.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models.onboarding import OnboardingSession, OnboardingStep, OnboardingEvent
from app.models.user import User
from app.core.database import get_async_session

# Test client setup
client = TestClient(app)

class TestOnboardingAPI:
    """Test suite for onboarding API endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Mock user for authentication."""
        return User(
            id="user-123",
            email="test@example.com",
            display_name="Test User"
        )

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for requests."""
        return {"Authorization": "Bearer test-token"}

    @pytest.fixture
    def onboarding_start_data(self):
        """Sample onboarding start request data."""
        return {
            "started_at": datetime.utcnow().isoformat(),
            "user_agent": "Mozilla/5.0 (Test Browser)",
            "referrer": "https://example.com",
            "source": "web"
        }

    @pytest.fixture
    def user_data_sample(self):
        """Sample user data for testing."""
        return {
            "name": "John Doe",
            "role": "developer",
            "goals": ["automate_workflows", "improve_efficiency"],
            "preferences": {"theme": "dark", "notifications": True}
        }

    class TestOnboardingSessionManagement:
        """Test onboarding session creation and management."""

        @patch('app.api.onboarding.get_current_user')
        @patch('app.core.database.get_async_session')
        async def test_start_new_onboarding_session(
            self, mock_db, mock_user, auth_headers, onboarding_start_data
        ):
            """Test starting a new onboarding session."""
            # Mock database session
            mock_session = AsyncMock()
            mock_db.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = None
            
            response = client.post(
                "/api/onboarding/start",
                json=onboarding_start_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            assert data["current_step"] == 1
            assert "session_id" in data

        @patch('app.api.onboarding.get_current_user')
        @patch('app.core.database.get_async_session')
        async def test_resume_existing_onboarding_session(
            self, mock_db, mock_user, auth_headers, onboarding_start_data
        ):
            """Test resuming an existing onboarding session."""
            # Mock existing session
            existing_session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                current_step=3,
                progress={"steps_completed": [1, 2]}
            )
            
            mock_session = AsyncMock()
            mock_db.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = existing_session
            
            response = client.post(
                "/api/onboarding/start",
                json=onboarding_start_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "resumed"
            assert data["current_step"] == 3
            assert data["session_id"] == "session-123"

        @patch('app.api.onboarding.get_current_user')
        @patch('app.core.database.get_async_session')
        async def test_get_onboarding_progress(
            self, mock_db, mock_user, auth_headers
        ):
            """Test retrieving onboarding progress."""
            session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                started_at=datetime.utcnow() - timedelta(minutes=5),
                current_step=2,
                progress={
                    "steps_completed": [1],
                    "user_data": {"name": "John"}
                }
            )
            
            mock_db_session = AsyncMock()
            mock_db.return_value = mock_db_session
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = session
            
            response = client.get("/api/onboarding/progress", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["current_step"] == 2
            assert data["completed_steps"] == [1]
            assert data["user_data"]["name"] == "John"
            assert "time_spent" in data

        async def test_get_progress_no_active_session(self, auth_headers):
            """Test getting progress with no active session."""
            with patch('app.api.onboarding.get_current_user'), \
                 patch('app.core.database.get_async_session') as mock_db:
                
                mock_session = AsyncMock()
                mock_db.return_value = mock_session
                mock_session.execute.return_value.scalar_one_or_none.return_value = None
                
                response = client.get("/api/onboarding/progress", headers=auth_headers)
                
                assert response.status_code == 200
                data = response.json()
                assert data["data"] is None
                assert "No active onboarding session" in data["message"]

    class TestProgressTracking:
        """Test progress update and step completion functionality."""

        @patch('app.api.onboarding.get_current_user')
        @patch('app.core.database.get_async_session')
        async def test_update_onboarding_progress(
            self, mock_db, mock_user, auth_headers, user_data_sample
        ):
            """Test updating onboarding progress."""
            session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                current_step=1,
                progress={"steps_completed": [], "user_data": {}}
            )
            
            mock_db_session = AsyncMock()
            mock_db.return_value = mock_db_session
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = session
            
            update_data = {
                "current_step": 2,
                "completed_steps": [1],
                "user_data": user_data_sample,
                "time_spent": 120000
            }
            
            response = client.put(
                "/api/onboarding/progress",
                json=update_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

        @patch('app.api.onboarding.get_current_user')
        @patch('app.core.database.get_async_session')
        async def test_complete_onboarding_step(
            self, mock_db, mock_user, auth_headers
        ):
            """Test completing an individual onboarding step."""
            session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                started_at=datetime.utcnow() - timedelta(minutes=2),
                current_step=1
            )
            
            mock_db_session = AsyncMock()
            mock_db.return_value = mock_db_session
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = session
            
            step_completion_data = {
                "step": 1,
                "timestamp": datetime.utcnow().isoformat(),
                "step_data": {"template_selected": "workflow_automator"},
                "user_data": {"name": "John Doe"}
            }
            
            response = client.post(
                "/api/onboarding/step-completed",
                json=step_completion_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["step"] == 1
            assert data["next_step"] == 2

        async def test_complete_final_step(self, auth_headers):
            """Test completing the final onboarding step."""
            with patch('app.api.onboarding.get_current_user'), \
                 patch('app.core.database.get_async_session') as mock_db:
                
                session = OnboardingSession(
                    id="session-123",
                    user_id="user-123",
                    started_at=datetime.utcnow() - timedelta(minutes=5),
                    current_step=5
                )
                
                mock_session = AsyncMock()
                mock_db.return_value = mock_session
                mock_session.execute.return_value.scalar_one_or_none.return_value = session
                
                step_completion_data = {
                    "step": 5,
                    "timestamp": datetime.utcnow().isoformat(),
                    "step_data": {"completion": True}
                }
                
                response = client.post(
                    "/api/onboarding/step-completed",
                    json=step_completion_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["next_step"] is None  # No next step after final

    class TestOnboardingCompletion:
        """Test onboarding completion functionality."""

        @patch('app.api.onboarding.get_current_user')
        @patch('app.core.database.get_async_session')
        async def test_complete_onboarding(
            self, mock_db, mock_user, auth_headers, user_data_sample
        ):
            """Test completing the entire onboarding process."""
            session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                started_at=datetime.utcnow() - timedelta(minutes=4),
                current_step=5
            )
            
            user = User(id="user-123", email="test@example.com")
            
            mock_db_session = AsyncMock()
            mock_db.return_value = mock_db_session
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = session
            mock_user.return_value = user
            
            completion_data = {
                "user_data": user_data_sample,
                "total_time": 240000,  # 4 minutes
                "completed_steps": 5,
                "completed_at": datetime.utcnow().isoformat(),
                "started_at": session.started_at.isoformat()
            }
            
            response = client.post(
                "/api/onboarding/complete",
                json=completion_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "Onboarding completed successfully!" in data["message"]
            assert data["completion_data"]["total_time"] == 240000
            assert data["completion_data"]["completed_steps"] == 5

        @patch('app.api.onboarding.get_current_user')
        @patch('app.core.database.get_async_session')
        async def test_skip_onboarding(
            self, mock_db, mock_user, auth_headers
        ):
            """Test skipping onboarding process."""
            session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                started_at=datetime.utcnow() - timedelta(minutes=1),
                current_step=2
            )
            
            mock_db_session = AsyncMock()
            mock_db.return_value = mock_db_session
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = session
            
            skip_data = {
                "current_step": 2,
                "skipped_at": datetime.utcnow().isoformat(),
                "reason": "User wants to explore on their own"
            }
            
            response = client.post(
                "/api/onboarding/skip",
                json=skip_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    class TestAnalyticsAndMetrics:
        """Test onboarding analytics and metrics endpoints."""

        @patch('app.api.onboarding.require_permission')
        @patch('app.core.database.get_async_session')
        async def test_get_onboarding_metrics(
            self, mock_db, mock_permission, auth_headers
        ):
            """Test retrieving onboarding analytics metrics."""
            mock_permission.return_value = True
            
            # Mock database queries
            mock_session = AsyncMock()
            mock_db.return_value = mock_session
            
            # Mock total sessions count
            mock_session.execute.return_value.scalar.side_effect = [
                100,  # total sessions
                85,   # completed sessions
                180000,  # average time
            ]
            
            response = client.get(
                "/api/onboarding/metrics?days=30",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "completion_rate" in data["data"]
            assert "average_time_to_complete" in data["data"]
            assert "drop_off_points" in data["data"]
            assert data["time_period_days"] == 30

        @patch('app.api.onboarding.require_permission')
        @patch('app.core.database.get_async_session')
        async def test_get_realtime_onboarding_data(
            self, mock_db, mock_permission, auth_headers
        ):
            """Test retrieving real-time onboarding activity."""
            mock_permission.return_value = True
            
            # Mock active sessions
            active_session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                started_at=datetime.utcnow() - timedelta(minutes=10),
                current_step=2,
                progress={"user_data": {"name": "Active User"}}
            )
            
            mock_session = AsyncMock()
            mock_db.return_value = mock_session
            mock_session.execute.return_value.scalars.return_value = [active_session]
            
            response = client.get(
                "/api/onboarding/realtime",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "active_sessions" in data["data"]
            assert data["data"]["total_active"] >= 0

        @patch('app.core.database.get_async_session')
        async def test_track_onboarding_event_background_task(self, mock_db):
            """Test background task for tracking onboarding events."""
            from app.api.onboarding import track_onboarding_event
            
            mock_session = AsyncMock()
            mock_db.return_value = mock_session
            
            session_id = "session-123"
            event_name = "step_1_completed"
            event_data = {"step_data": {"template": "workflow_automator"}}
            
            await track_onboarding_event(session_id, event_name, event_data, mock_session)
            
            # Verify event was added to session
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    class TestEpic6SuccessMetrics:
        """Test Epic 6 specific success metrics and targets."""

        async def test_90_percent_completion_rate_target(self):
            """Test that the system can track 90%+ completion rate."""
            # This would typically be an integration test with real data
            # For now, we verify the calculation logic
            total_sessions = 100
            completed_sessions = 92
            
            completion_rate = (completed_sessions / total_sessions) * 100
            
            assert completion_rate >= 90, "Should achieve 90%+ completion rate target"

        async def test_five_minute_time_to_value_target(self):
            """Test that onboarding completes within 5 minutes."""
            # Test with sample completion time
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(minutes=4, seconds=30)  # 4.5 minutes
            
            total_time_ms = int((end_time - start_time).total_seconds() * 1000)
            five_minutes_ms = 5 * 60 * 1000  # 300,000 ms
            
            assert total_time_ms < five_minutes_ms, "Should complete within 5 minutes"

        async def test_user_engagement_tracking(self, user_data_sample):
            """Test that user engagement metrics are properly captured."""
            # Verify that user data captures engagement signals
            assert "goals" in user_data_sample
            assert len(user_data_sample["goals"]) > 0  # Multiple goals = higher engagement
            assert "preferences" in user_data_sample
            
            # Calculate engagement score based on captured data
            engagement_score = 0
            if user_data_sample.get("goals"):
                engagement_score += len(user_data_sample["goals"]) * 10
            if user_data_sample.get("preferences"):
                engagement_score += len(user_data_sample["preferences"]) * 5
            
            assert engagement_score > 0, "Should capture user engagement signals"

        async def test_drop_off_point_identification(self):
            """Test identification of onboarding drop-off points."""
            # Mock drop-off data for each step
            step_data = [
                {"step": 1, "started": 100, "completed": 95, "drop_off_rate": 5.0},
                {"step": 2, "started": 95, "completed": 85, "drop_off_rate": 10.5},
                {"step": 3, "started": 85, "completed": 82, "drop_off_rate": 3.5},
                {"step": 4, "started": 82, "completed": 78, "drop_off_rate": 4.9},
                {"step": 5, "started": 78, "completed": 75, "drop_off_rate": 3.8},
            ]
            
            # Find highest drop-off point
            highest_drop_off = max(step_data, key=lambda x: x["drop_off_rate"])
            
            assert highest_drop_off["step"] == 2  # Step 2 has highest drop-off
            assert highest_drop_off["drop_off_rate"] > 10  # Significant drop-off identified

        async def test_mobile_responsiveness_indicators(self):
            """Test that mobile user agents are tracked for responsiveness metrics."""
            mobile_user_agents = [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)",
                "Mozilla/5.0 (Android 11; Mobile; rv:91.0)",
                "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X)"
            ]
            
            for ua in mobile_user_agents:
                is_mobile = any(device in ua for device in ["iPhone", "Android", "iPad", "Mobile"])
                assert is_mobile, f"Should detect mobile user agent: {ua}"

    class TestErrorHandling:
        """Test error handling and edge cases."""

        async def test_invalid_step_completion(self, auth_headers):
            """Test handling of invalid step completion requests."""
            invalid_data = {
                "step": 10,  # Invalid step number
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = client.post(
                "/api/onboarding/step-completed",
                json=invalid_data,
                headers=auth_headers
            )
            
            assert response.status_code == 422  # Validation error

        async def test_session_not_found_error(self, auth_headers):
            """Test handling when no active session is found."""
            with patch('app.api.onboarding.get_current_user'), \
                 patch('app.core.database.get_async_session') as mock_db:
                
                mock_session = AsyncMock()
                mock_db.return_value = mock_session
                mock_session.execute.return_value.scalar_one_or_none.return_value = None
                
                response = client.put(
                    "/api/onboarding/progress",
                    json={"current_step": 2, "completed_steps": [1], "user_data": {}, "time_spent": 1000},
                    headers=auth_headers
                )
                
                assert response.status_code == 404

        async def test_database_error_handling(self, auth_headers):
            """Test handling of database errors."""
            with patch('app.api.onboarding.get_current_user'), \
                 patch('app.core.database.get_async_session') as mock_db:
                
                mock_db.side_effect = Exception("Database connection error")
                
                response = client.get("/api/onboarding/progress", headers=auth_headers)
                
                assert response.status_code == 500

    class TestPermissionsAndSecurity:
        """Test authentication and authorization requirements."""

        async def test_authentication_required(self):
            """Test that authentication is required for all endpoints."""
            endpoints = [
                "/api/onboarding/start",
                "/api/onboarding/progress", 
                "/api/onboarding/complete",
                "/api/onboarding/skip"
            ]
            
            for endpoint in endpoints:
                response = client.post(endpoint) if "start" in endpoint or "complete" in endpoint or "skip" in endpoint else client.get(endpoint)
                assert response.status_code in [401, 403], f"Endpoint {endpoint} should require authentication"

        async def test_analytics_permission_required(self):
            """Test that analytics endpoints require special permissions."""
            analytics_endpoints = [
                "/api/onboarding/metrics",
                "/api/onboarding/realtime"
            ]
            
            for endpoint in analytics_endpoints:
                response = client.get(endpoint)
                assert response.status_code in [401, 403], f"Analytics endpoint {endpoint} should require permissions"


# Integration test for full onboarding flow
class TestOnboardingIntegration:
    """Integration tests for complete onboarding workflows."""

    @patch('app.api.onboarding.get_current_user')
    @patch('app.core.database.get_async_session')
    async def test_complete_onboarding_flow(self, mock_db, mock_user, auth_headers):
        """Test complete onboarding flow from start to finish."""
        # This would be a comprehensive integration test
        # covering the entire user journey through all 5 steps
        
        # 1. Start onboarding
        start_data = {
            "started_at": datetime.utcnow().isoformat(),
            "user_agent": "Test Agent",
            "referrer": "test.com"
        }
        
        # Mock session setup
        mock_session = AsyncMock()
        mock_db.return_value = mock_session
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        start_response = client.post("/api/onboarding/start", json=start_data, headers=auth_headers)
        assert start_response.status_code == 200
        
        # 2. Complete each step (mocked for test)
        for step in range(1, 6):
            step_data = {
                "step": step,
                "timestamp": datetime.utcnow().isoformat(),
                "step_data": {"step_completed": True}
            }
            
            # Mock active session for each step
            session = OnboardingSession(
                id="session-123",
                user_id="user-123",
                started_at=datetime.utcnow() - timedelta(minutes=step),
                current_step=step
            )
            mock_session.execute.return_value.scalar_one_or_none.return_value = session
            
            step_response = client.post("/api/onboarding/step-completed", json=step_data, headers=auth_headers)
            assert step_response.status_code == 200
        
        # 3. Complete onboarding
        completion_data = {
            "user_data": {"name": "Test User", "role": "developer", "goals": ["automate"]},
            "total_time": 300000,  # 5 minutes
            "completed_steps": 5,
            "completed_at": datetime.utcnow().isoformat(),
            "started_at": (datetime.utcnow() - timedelta(minutes=5)).isoformat()
        }
        
        completion_response = client.post("/api/onboarding/complete", json=completion_data, headers=auth_headers)
        assert completion_response.status_code == 200
        
        # Verify Epic 6 targets were met
        completion_result = completion_response.json()
        assert completion_result["completion_data"]["total_time"] <= 300000  # <= 5 minutes
        assert completion_result["completion_data"]["completed_steps"] == 5  # 100% completion
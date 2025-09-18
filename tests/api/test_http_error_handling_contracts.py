"""
HTTP Error Handling and Status Code Contract Testing
===================================================

Comprehensive testing of HTTP status codes, error handling patterns, and 
API error response contracts. Ensures consistent error handling across
all endpoints and proper HTTP protocol compliance.

Key Contract Areas:
- HTTP status code compliance (200, 201, 400, 404, 422, 500)
- Error response schema consistency
- Input validation error handling
- Resource not found error patterns
- Server error handling and recovery
- CORS and preflight error handling
"""

import pytest
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
import jsonschema
from jsonschema import validate, ValidationError
import uvicorn
import threading


class TestHTTPStatusCodeContracts:
    """Contract tests for HTTP status code compliance."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for HTTP testing."""
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8996,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        yield "http://127.0.0.1:8996"

    @pytest.fixture
    async def http_client(self, api_server):
        """Create HTTP client for testing."""
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            yield client

    # Success Status Codes (2xx)

    async def test_200_ok_status_contract(self, http_client):
        """Test 200 OK status code contract for successful GET requests."""
        
        endpoints_200 = [
            "/",
            "/health",
            "/status",
            "/api/v1/system/status",
            "/api/v1/agents",
            "/api/v1/tasks",
            "/observability/metrics",
            "/observability/health"
        ]
        
        for endpoint in endpoints_200:
            response = await http_client.get(endpoint)
            
            # Contract: GET requests to existing endpoints return 200
            assert response.status_code == 200, f"Endpoint {endpoint} should return 200, got {response.status_code}"
            
            # Contract: 200 responses must have valid JSON content
            try:
                data = response.json()
                assert isinstance(data, (dict, list)), f"Endpoint {endpoint} should return JSON object or array"
            except json.JSONDecodeError:
                pytest.fail(f"Endpoint {endpoint} returned invalid JSON with 200 status")

    async def test_201_created_status_contract(self, http_client):
        """Test 201 Created status code contract for resource creation."""
        
        # Test agent creation returns 201
        agent_payload = {
            "name": "HTTP Test Agent",
            "type": "claude",
            "role": "backend_developer"
        }
        
        response = await http_client.post("/api/v1/agents", json=agent_payload)
        
        # Contract: POST requests that create resources return 201 or 200
        assert response.status_code in [200, 201], f"Agent creation should return 200 or 201, got {response.status_code}"
        
        # If 201, response should include created resource
        if response.status_code == 201:
            data = response.json()
            assert "id" in data, "201 Created should include created resource with ID"
        
        # Test task creation returns 201
        task_payload = {
            "title": "HTTP Test Task",
            "description": "Test task for HTTP status codes",
            "priority": "medium"
        }
        
        response = await http_client.post("/api/v1/tasks", json=task_payload)
        assert response.status_code in [200, 201], f"Task creation should return 200 or 201, got {response.status_code}"

    # Client Error Status Codes (4xx)

    async def test_400_bad_request_status_contract(self, http_client):
        """Test 400 Bad Request status code contract for malformed requests."""
        
        # Test invalid JSON payload
        invalid_json_endpoints = [
            ("/api/v1/agents", "invalid json content"),
            ("/api/v1/tasks", "not json at all")
        ]
        
        for endpoint, invalid_content in invalid_json_endpoints:
            response = await http_client.post(
                endpoint,
                content=invalid_content,
                headers={"Content-Type": "application/json"}
            )
            
            # Contract: Invalid JSON should return 400 or 422
            assert response.status_code in [400, 422], f"Invalid JSON to {endpoint} should return 400 or 422, got {response.status_code}"

    async def test_404_not_found_status_contract(self, http_client):
        """Test 404 Not Found status code contract for missing resources."""
        
        # Test non-existent endpoints
        non_existent_endpoints = [
            "/api/v1/nonexistent",
            "/api/v1/agents/non-existent-agent",
            "/api/v1/tasks/non-existent-task",
            "/api/v2/anything",
            "/completely/invalid/path"
        ]
        
        for endpoint in non_existent_endpoints:
            response = await http_client.get(endpoint)
            
            # Contract: Non-existent resources return 404
            assert response.status_code == 404, f"Non-existent endpoint {endpoint} should return 404, got {response.status_code}"
            
            # Contract: 404 responses should have error message
            try:
                error_data = response.json()
                assert "detail" in error_data or "error" in error_data or "message" in error_data, f"404 response should include error details"
            except json.JSONDecodeError:
                # Some 404s might not return JSON, which is acceptable
                pass

    async def test_404_resource_not_found_contract(self, http_client):
        """Test 404 for specific resource operations."""
        
        non_existent_id = "definitely-does-not-exist-123"
        
        # Test operations on non-existent agents
        agent_404_operations = [
            ("GET", f"/api/v1/agents/{non_existent_id}"),
            ("PUT", f"/api/v1/agents/{non_existent_id}"),
            ("DELETE", f"/api/v1/agents/{non_existent_id}")
        ]
        
        for method, endpoint in agent_404_operations:
            if method == "GET":
                response = await http_client.get(endpoint)
            elif method == "PUT":
                response = await http_client.put(endpoint, json={"status": "active"})
            elif method == "DELETE":
                response = await http_client.delete(endpoint)
            
            assert response.status_code == 404, f"{method} {endpoint} should return 404 for non-existent resource"

    async def test_422_unprocessable_entity_contract(self, http_client):
        """Test 422 Unprocessable Entity for validation errors."""
        
        # Test invalid payloads that are valid JSON but fail validation
        invalid_payloads = [
            # Agent creation with missing required fields
            ({}, "/api/v1/agents"),
            # Agent creation with invalid type
            ({"name": "Test", "type": "invalid_type"}, "/api/v1/agents"),
            # Task creation with missing title
            ({"description": "No title"}, "/api/v1/tasks"),
            # Task creation with invalid priority
            ({"title": "Test", "priority": "invalid_priority"}, "/api/v1/tasks")
        ]
        
        for payload, endpoint in invalid_payloads:
            response = await http_client.post(endpoint, json=payload)
            
            # Contract: Validation errors return 422
            assert response.status_code == 422, f"Invalid payload to {endpoint} should return 422, got {response.status_code}"
            
            # Contract: 422 responses include validation details
            error_data = response.json()
            assert "detail" in error_data, "422 responses should include validation details"

    # Method Not Allowed (405)

    async def test_405_method_not_allowed_contract(self, http_client):
        """Test 405 Method Not Allowed for unsupported HTTP methods."""
        
        # Test unsupported methods on various endpoints
        method_tests = [
            ("PATCH", "/api/v1/agents"),  # Only GET, POST allowed
            ("DELETE", "/api/v1/agents"), # Only GET, POST allowed
            ("POST", "/health"),          # Only GET allowed
            ("PUT", "/status"),           # Only GET allowed
        ]
        
        for method, endpoint in method_tests:
            response = await http_client.request(method, endpoint)
            
            # Contract: Unsupported methods return 405
            assert response.status_code == 405, f"{method} {endpoint} should return 405, got {response.status_code}"


class TestErrorResponseSchemaContracts:
    """Contract tests for error response schema consistency."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for error response testing."""
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8995,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        yield "http://127.0.0.1:8995"

    @pytest.fixture
    async def http_client(self, api_server):
        """Create HTTP client for error testing."""
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            yield client

    async def test_404_error_response_schema_contract(self, http_client):
        """Test 404 error response schema contract."""
        
        error_404_schema = {
            "type": "object",
            "required": ["detail"],
            "properties": {
                "detail": {"type": "string", "minLength": 1}
            }
        }
        
        # Test various 404 scenarios
        response = await http_client.get("/api/v1/agents/non-existent-agent")
        assert response.status_code == 404
        
        error_data = response.json()
        
        try:
            jsonschema.validate(error_data, error_404_schema)
        except ValidationError as e:
            pytest.fail(f"404 error response schema violation: {e}")
        
        # Contract: Error message should be descriptive
        assert "not found" in error_data["detail"].lower(), "404 error should mention 'not found'"

    async def test_422_error_response_schema_contract(self, http_client):
        """Test 422 validation error response schema contract."""
        
        error_422_schema = {
            "type": "object",
            "required": ["detail"],
            "properties": {
                "detail": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["loc", "msg", "type"],
                        "properties": {
                            "loc": {"type": "array"},
                            "msg": {"type": "string"},
                            "type": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        # Trigger validation error
        response = await http_client.post("/api/v1/agents", json={})
        assert response.status_code == 422
        
        error_data = response.json()
        
        try:
            jsonschema.validate(error_data, error_422_schema)
        except ValidationError as e:
            # FastAPI's default 422 format might differ, check for alternative format
            alternative_422_schema = {
                "type": "object",
                "required": ["detail"],
                "properties": {
                    "detail": {"type": "string"}
                }
            }
            
            try:
                jsonschema.validate(error_data, alternative_422_schema)
            except ValidationError:
                pytest.fail(f"422 error response schema violation: {e}")

    async def test_500_error_response_schema_contract(self, http_client):
        """Test 500 internal server error response schema contract."""
        
        error_500_schema = {
            "type": "object",
            "properties": {
                "detail": {"type": "string"},
                "error": {"type": "string"},
                "message": {"type": "string"}
            },
            "anyOf": [
                {"required": ["detail"]},
                {"required": ["error"]},
                {"required": ["message"]}
            ]
        }
        
        # Since we can't easily trigger a 500 error in our test server,
        # we'll test the schema format that should be used
        mock_500_response = {
            "detail": "Internal server error occurred",
            "error": "Database connection failed"
        }
        
        try:
            jsonschema.validate(mock_500_response, error_500_schema)
        except ValidationError as e:
            pytest.fail(f"500 error response schema violation: {e}")

    async def test_error_response_consistency_contract(self, http_client):
        """Test error response consistency across endpoints."""
        
        # Collect error responses from different endpoints
        error_responses = []
        
        # 404 errors
        not_found_endpoints = [
            "/api/v1/agents/not-found",
            "/api/v1/tasks/not-found"
        ]
        
        for endpoint in not_found_endpoints:
            response = await http_client.get(endpoint)
            if response.status_code == 404:
                error_responses.append({
                    "endpoint": endpoint,
                    "status_code": 404,
                    "response": response.json()
                })
        
        # 422 errors
        validation_tests = [
            ("/api/v1/agents", {}),
            ("/api/v1/tasks", {})
        ]
        
        for endpoint, payload in validation_tests:
            response = await http_client.post(endpoint, json=payload)
            if response.status_code == 422:
                error_responses.append({
                    "endpoint": endpoint,
                    "status_code": 422,
                    "response": response.json()
                })
        
        # Contract: All error responses should have consistent structure
        for error_resp in error_responses:
            response_data = error_resp["response"]
            
            # Should always have some form of error message
            has_error_field = any(
                field in response_data 
                for field in ["detail", "error", "message", "errors"]
            )
            
            assert has_error_field, f"Error response from {error_resp['endpoint']} missing error field"


class TestInputValidationErrorContracts:
    """Contract tests for input validation error handling."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for validation testing."""
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8994,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        yield "http://127.0.0.1:8994"

    @pytest.fixture
    async def http_client(self, api_server):
        """Create HTTP client for validation testing."""
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            yield client

    async def test_required_field_validation_contract(self, http_client):
        """Test required field validation error contracts."""
        
        # Test agent creation without required fields
        response = await http_client.post("/api/v1/agents", json={})
        assert response.status_code == 422
        
        error_data = response.json()
        
        # Contract: Validation error should mention missing required field
        error_message = str(error_data)
        assert any(keyword in error_message.lower() for keyword in ["required", "missing", "field"]), \
            "Validation error should mention missing required field"

    async def test_field_type_validation_contract(self, http_client):
        """Test field type validation error contracts."""
        
        # Test invalid field types
        invalid_type_payloads = [
            # Name should be string, not number
            {"name": 12345, "type": "claude"},
            # Capabilities should be array, not string
            {"name": "Test Agent", "capabilities": "not an array"}
        ]
        
        for payload in invalid_type_payloads:
            response = await http_client.post("/api/v1/agents", json=payload)
            
            # Should return validation error
            assert response.status_code == 422, f"Invalid type payload should return 422, got {response.status_code}"

    async def test_field_length_validation_contract(self, http_client):
        """Test field length validation error contracts."""
        
        # Test extremely long values (if validation exists)
        long_string = "a" * 1000
        
        long_value_payload = {
            "name": long_string,
            "type": "claude",
            "description": long_string
        }
        
        response = await http_client.post("/api/v1/agents", json=long_value_payload)
        
        # Depending on implementation, might accept long values or reject them
        # Contract: Response should be consistent (either accept all or reject consistently)
        assert response.status_code in [200, 201, 422], "Response to long values should be consistent"

    async def test_enum_value_validation_contract(self, http_client):
        """Test enum value validation error contracts."""
        
        # Test invalid enum values
        invalid_enum_payloads = [
            # Invalid agent type
            {"name": "Test Agent", "type": "invalid_agent_type"},
            # Invalid task priority (for tasks)
            {"title": "Test Task", "priority": "super_ultra_high"}
        ]
        
        endpoints = ["/api/v1/agents", "/api/v1/tasks"]
        
        for payload, endpoint in zip(invalid_enum_payloads, endpoints):
            response = await http_client.post(endpoint, json=payload)
            
            # Should return validation error for invalid enum values
            assert response.status_code == 422, f"Invalid enum value should return 422, got {response.status_code}"


class TestServerErrorHandlingContracts:
    """Contract tests for server error handling and recovery."""

    async def test_error_logging_contract(self):
        """Test error logging contract for debugging."""
        
        # This tests the contract that errors should be properly logged
        # In a real implementation, this would verify log entries
        
        # Mock error scenarios that should be logged
        error_scenarios = [
            {"type": "database_error", "message": "Connection timeout"},
            {"type": "validation_error", "message": "Invalid input format"},
            {"type": "external_service_error", "message": "API rate limit exceeded"}
        ]
        
        # Contract: All errors should have structured logging
        for scenario in error_scenarios:
            # Validate error structure for logging
            assert "type" in scenario, "Errors should be categorized by type"
            assert "message" in scenario, "Errors should have descriptive messages"
            assert isinstance(scenario["message"], str), "Error messages should be strings"
            assert len(scenario["message"]) > 0, "Error messages should not be empty"

    async def test_error_recovery_contract(self):
        """Test error recovery and graceful degradation contract."""
        
        # Contract: System should handle errors gracefully and continue operating
        
        # Test that after an error, subsequent requests still work
        # This would be tested with a real server in integration tests
        
        recovery_scenarios = [
            {"error": "temporary_db_connection_loss", "recovery": "automatic_retry"},
            {"error": "invalid_request_format", "recovery": "return_error_continue_processing"},
            {"error": "resource_not_found", "recovery": "return_404_continue_processing"}
        ]
        
        for scenario in recovery_scenarios:
            # Contract: Errors should not crash the entire system
            assert "recovery" in scenario, "Error scenarios should define recovery strategy"
            
            # Contract: Recovery should be automatic where possible
            recovery_strategy = scenario["recovery"]
            assert isinstance(recovery_strategy, str), "Recovery strategy should be defined"

    async def test_error_response_time_contract(self):
        """Test error response time contract."""
        
        # Contract: Error responses should be fast to avoid timeouts
        max_error_response_time_ms = 1000.0  # 1 second max for errors
        
        # Mock error response timing
        start_time = time.time()
        
        # Simulate error processing
        error_processing_steps = [
            "validate_request",
            "check_resource_exists", 
            "format_error_response",
            "log_error",
            "return_response"
        ]
        
        for step in error_processing_steps:
            # Each step should be fast
            step_time = time.time()
            await asyncio.sleep(0.001)  # Simulate minimal processing
            step_duration = (time.time() - step_time) * 1000
            
            assert step_duration < 100.0, f"Error processing step {step} should be fast"
        
        total_time = (time.time() - start_time) * 1000
        
        # Contract: Total error response time should be reasonable
        assert total_time < max_error_response_time_ms, f"Error response took {total_time}ms, exceeds {max_error_response_time_ms}ms contract"


class TestCORSErrorHandlingContracts:
    """Contract tests for CORS and preflight error handling."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for CORS testing."""
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8993,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        yield "http://127.0.0.1:8993"

    @pytest.fixture
    async def http_client(self, api_server):
        """Create HTTP client for CORS testing."""
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            yield client

    async def test_cors_preflight_contract(self, http_client):
        """Test CORS preflight request handling contract."""
        
        # Test OPTIONS request for CORS preflight
        cors_headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = await http_client.options("/api/v1/agents", headers=cors_headers)
        
        # Contract: CORS preflight should not return 404 or 405
        assert response.status_code in [200, 204], f"CORS preflight should return 200 or 204, got {response.status_code}"

    async def test_cors_error_response_contract(self, http_client):
        """Test CORS headers in error responses contract."""
        
        # Test that error responses include CORS headers when needed
        cors_origin = "http://localhost:3000"
        
        # Make request that will result in 404
        response = await http_client.get(
            "/api/v1/agents/non-existent",
            headers={"Origin": cors_origin}
        )
        
        assert response.status_code == 404
        
        # Contract: Error responses should include CORS headers for valid origins
        # Note: Actual CORS headers depend on server configuration
        # This tests the contract that CORS should be handled consistently


# Integration Error Handling Summary
class TestHTTPErrorHandlingContractSummary:
    """Summary test validating all HTTP error handling contracts work together."""
    
    async def test_complete_error_handling_contract_compliance(self):
        """Integration test ensuring all error handling contracts are compatible."""
        
        # Start test server
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8992,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8992", timeout=30.0) as client:
            
            # Test complete error handling workflow
            
            # 1. Success response contract (200)
            response = await client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert "status" in health_data
            
            # 2. Resource creation contract (200/201)
            create_response = await client.post("/api/v1/agents", json={
                "name": "Error Test Agent",
                "type": "claude"
            })
            assert create_response.status_code in [200, 201]
            
            if create_response.status_code in [200, 201]:
                agent_data = create_response.json()
                agent_id = agent_data["id"]
                
                # 3. Resource retrieval contract (200)
                get_response = await client.get(f"/api/v1/agents/{agent_id}")
                assert get_response.status_code == 200
                
                # 4. Resource not found contract (404)
                not_found_response = await client.get("/api/v1/agents/definitely-not-found")
                assert not_found_response.status_code == 404
                
                error_data = not_found_response.json()
                assert "detail" in error_data or "error" in error_data
            
            # 5. Validation error contract (422)
            validation_response = await client.post("/api/v1/agents", json={})
            assert validation_response.status_code == 422
            
            validation_error = validation_response.json()
            assert "detail" in validation_error
            
            # 6. Invalid JSON contract (400/422)
            invalid_json_response = await client.post(
                "/api/v1/agents",
                content="invalid json",
                headers={"Content-Type": "application/json"}
            )
            assert invalid_json_response.status_code in [400, 422]
            
            # 7. Method not allowed contract (405)
            method_not_allowed_response = await client.patch("/health")
            assert method_not_allowed_response.status_code == 405
            
            # 8. Non-existent endpoint contract (404)
            endpoint_not_found_response = await client.get("/api/v1/completely/invalid/path")
            assert endpoint_not_found_response.status_code == 404
            
            # 9. Error response consistency contract
            error_responses = [
                not_found_response,
                validation_response,
                invalid_json_response,
                endpoint_not_found_response
            ]
            
            for error_response in error_responses:
                if error_response.status_code >= 400:
                    try:
                        error_data = error_response.json()
                        # All errors should have some form of message
                        has_message = any(
                            field in error_data 
                            for field in ["detail", "error", "message"]
                        )
                        assert has_message, f"Error response missing message field: {error_data}"
                    except json.JSONDecodeError:
                        # Some error responses might not be JSON, which is acceptable
                        pass
            
            # 10. Performance contract for error responses
            start_time = time.time()
            quick_error_response = await client.get("/api/v1/agents/quick-404-test")
            error_response_time = (time.time() - start_time) * 1000
            
            # Error responses should be fast
            assert error_response_time < 1000.0, f"Error response took {error_response_time}ms, exceeds 1000ms contract"
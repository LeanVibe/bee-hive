"""
CLI Testing Package for LeanVibe Agent Hive 2.0

This package contains comprehensive CLI testing implementations following
the strategy outlined in CLI_TESTING_STRATEGY.md.

Test modules:
- test_cli_units_foundation.py: Unit tests for CLI command parsing and logic
- test_cli_integration_mocked.py: Integration tests with mocked API responses  
- test_cli_e2e_integration.py: End-to-end tests with running API server
- test_cli_workflows.py: Complete user workflow testing
- test_cli_performance_errors.py: Performance and error handling tests
"""

# CLI test markers for selective test execution
CLI_UNIT_TESTS = "cli_units"
CLI_INTEGRATION_TESTS = "cli_integration" 
CLI_E2E_TESTS = "cli_e2e"
CLI_WORKFLOW_TESTS = "cli_workflows"
CLI_PERFORMANCE_TESTS = "cli_performance"
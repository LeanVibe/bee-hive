"""
Integration Testing - Component Integration Tests

This directory contains integration tests that test how components work together
without full system integration. This is Level 3 of our testing pyramid.

Test Structure:
- workflows/ - Contains workflow-specific integration tests
- Each workflow test uses real components but controlled test data
- Tests verify message passing between components
- Validates state changes across component boundaries
- Tests failure scenarios and recovery mechanisms
- Ensures data consistency across component interactions

Key Integration Points Tested:
- Orchestrator ↔ Agent communication flows
- Task creation → Task execution workflows  
- WebSocket ↔ Redis pub/sub message routing
- Database ↔ Cache synchronization
- Configuration → Component initialization
"""
"""
Unit Testing - Component Isolation Tests

This directory contains unit tests for individual components tested in complete isolation
with all external dependencies mocked. This is Level 2 of our testing pyramid.

Test Structure:
- Each component has its own test file
- All external dependencies are mocked using AsyncMock
- Tests verify component behavior without side effects
- Focus on business logic and error handling
"""
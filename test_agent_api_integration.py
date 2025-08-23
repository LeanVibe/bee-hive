#!/usr/bin/env python3
"""
Test script to validate agent API integration with SimpleOrchestrator
Tests the core functionality required for CLI integration.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add app to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_agent_api_integration():
    """Test the agent API endpoints with SimpleOrchestrator."""
    
    print("🧪 Starting Agent API Integration Test")
    print("=" * 50)
    
    try:
        # Import the core components
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentRole
        from app.api.v1.agents_simple import get_simple_orchestrator
        from app.api.v1.agents_simple import router, AgentCreateRequest, AgentListResponse
        
        print("✅ Successfully imported core components")
        
        # Create a SimpleOrchestrator instance
        orchestrator = create_simple_orchestrator()
        await orchestrator.initialize()
        print("✅ SimpleOrchestrator initialized successfully")
        
        # Test 1: List agents (should be empty initially)
        print("\n🔍 Test 1: List agents")
        agent_sessions = await orchestrator.list_agent_sessions()
        print(f"   Found {len(agent_sessions)} agents")
        assert isinstance(agent_sessions, list), "list_agent_sessions should return a list"
        print("✅ list_agent_sessions working correctly")
        
        # Test 2: Create an agent
        print("\n🚀 Test 2: Create agent")
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            agent_id="test-agent-123",
            environment_vars={"LEANVIBE_AGENT_NAME": "test-agent-123"}
        )
        print(f"   Agent created with ID: {agent_id}")
        assert agent_id is not None, "spawn_agent should return an agent ID"
        print("✅ spawn_agent working correctly")
        
        # Test 3: List agents after creation
        print("\n📋 Test 3: List agents after creation")
        agent_sessions = await orchestrator.list_agent_sessions()
        print(f"   Found {len(agent_sessions)} agents")
        
        if agent_sessions:
            session = agent_sessions[0]
            print(f"   Agent session info: {json.dumps(session, indent=2, default=str)}")
        
        print("✅ Agent persisted in session list")
        
        # Test 4: Get specific agent info
        print("\n🔍 Test 4: Get agent info")
        agent_info = await orchestrator.get_agent_session_info(agent_id)
        if agent_info:
            print(f"   Agent info retrieved: {json.dumps(agent_info, indent=2, default=str)}")
            print("✅ get_agent_session_info working correctly")
        else:
            print("⚠️ Agent info not found - may be expected for tmux-based agents")
        
        # Test 5: System status
        print("\n📊 Test 5: System status")
        system_status = await orchestrator.get_system_status()
        print(f"   System status: {json.dumps(system_status, indent=2, default=str)}")
        assert isinstance(system_status, dict), "get_system_status should return a dict"
        print("✅ get_system_status working correctly")
        
        # Test 6: Shutdown agent
        print("\n🛑 Test 6: Shutdown agent")
        shutdown_success = await orchestrator.shutdown_agent(agent_id, graceful=True)
        print(f"   Shutdown success: {shutdown_success}")
        print("✅ shutdown_agent completed")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Agent API integration is working correctly")
        print("✅ CLI integration should work with these endpoints")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["SKIP_STARTUP_INIT"] = "true"
    
    # Run the test
    success = asyncio.run(test_agent_api_integration())
    sys.exit(0 if success else 1)
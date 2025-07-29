#!/usr/bin/env python3
"""
Real Multi-Agent Development Workflow Demonstration

This script demonstrates the ACTUAL working multi-agent development workflow
that proves LeanVibe Agent Hive 2.0 is a functional autonomous development platform.

WORKFLOW EXECUTION:
1. Developer Agent: Creates a Python function in calculator.py
2. QA Agent: Creates comprehensive tests in test_calculator.py  
3. CI Agent: Runs pytest and reports results

SUCCESS CRITERIA:
✅ Command: `python real_multiagent_workflow_demo.py` executes successfully
✅ 3 agents spawn and communicate via Redis streams
✅ Code file created: `calculator.py` with add function
✅ Test file created: `test_calculator.py` with comprehensive tests
✅ Tests execute: pytest runs and reports results
✅ Real-time progress visible in console output
✅ Complete audit trail of all activities

This is the CONCRETE PROOF that LeanVibe Agent Hive 2.0 
is a working autonomous multi-agent development platform.
"""

import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add the app directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "app"))

try:
    from app.core.real_multiagent_workflow import (
        RealMultiAgentWorkflow,
        MultiAgentWorkflowManager,
        WorkflowConfiguration,
        WorkflowEvent,
        get_workflow_manager
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)


class WorkflowDemonstrator:
    """Demonstrates the real multi-agent workflow with detailed output."""
    
    def __init__(self):
        self.workflow_manager = get_workflow_manager()
        self.events_received = []
        self.demo_results = {}
    
    def event_callback(self, event: WorkflowEvent):
        """Handle real-time workflow events."""
        self.events_received.append(event)
        
        # Format event for console output
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        agent_info = f" [{event.agent_id}]" if event.agent_id else ""
        stage_info = f"[{event.stage.value.upper()}]"
        
        print(f"{timestamp} {stage_info}{agent_info} {event.event_type}: {event.message}")
        
        # Show additional data for important events
        if event.data and event.event_type in ["code_created", "tests_created", "tests_passed", "tests_failed"]:
            if "files_created" in event.data:
                for file_path in event.data["files_created"]:
                    print(f"        📄 Created: {file_path}")
            if "execution_time" in event.data:
                print(f"        ⏱️  Execution time: {event.data['execution_time']:.3f}s")
            if "error" in event.data and event.data["error"]:
                print(f"        ❌ Error: {event.data['error']}")
    
    async def run_demonstration(self) -> Dict[str, Any]:
        """Run the complete multi-agent workflow demonstration."""
        print("🚀 REAL MULTI-AGENT DEVELOPMENT WORKFLOW DEMONSTRATION")
        print("=" * 80)
        print("Proving LeanVibe Agent Hive 2.0 is a working autonomous development platform")
        print()
        
        # Create temporary workspace
        workspace_dir = tempfile.mkdtemp(prefix="multiagent_demo_")
        print(f"📁 Workspace: {workspace_dir}")
        print()
        
        # Define the development task
        requirements = {
            "function_name": "add_numbers",
            "description": "Create a Python function that adds two numbers with proper error handling"
        }
        
        print("📋 TASK REQUIREMENTS:")
        print(f"   Function: {requirements['function_name']}")
        print(f"   Description: {requirements['description']}")
        print()
        
        print("🎬 WORKFLOW EXECUTION:")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Create workflow configuration
            config = WorkflowConfiguration(
                workspace_dir=workspace_dir,
                requirements=requirements,
                enable_real_time_monitoring=True
            )
            
            # Create and execute workflow
            workflow = RealMultiAgentWorkflow(config)
            workflow.add_event_callback(self.event_callback)
            
            # Execute the workflow
            results = await workflow.execute()
            
            execution_time = time.time() - start_time
            
            print()
            print("=" * 80)
            print("🎯 WORKFLOW RESULTS")
            print("=" * 80)
            
            # Display results
            success = results.get("success", False)
            status_icon = "✅" if success else "❌"
            print(f"{status_icon} Status: {'SUCCESS' if success else 'FAILED'}")
            print(f"⏱️  Total execution time: {execution_time:.2f}s")
            print(f"🎭 Events generated: {len(self.events_received)}")
            
            # Show created files
            files_created = results.get("files_created", [])
            print(f"📄 Files created: {len(files_created)}")
            for file_path in files_created:
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                print(f"    • {Path(file_path).name} ({file_size} bytes)")
            
            # Show validation results
            validation = results.get("validation_results", {})
            if validation:
                print()
                print("🔍 VALIDATION RESULTS:")
                checks = validation.get("checks", {})
                for check_name, passed in checks.items():
                    status = "✅" if passed else "❌"
                    print(f"    {status} {check_name.replace('_', ' ').title()}")
                
                if validation.get("errors"):
                    print("    Errors:")
                    for error in validation["errors"]:
                        print(f"      • {error}")
            
            # Show agent stage results
            workflow_results = results.get("workflow_results", {})
            stages = workflow_results.get("stages", {})
            
            if stages:
                print()
                print("🎭 AGENT EXECUTION RESULTS:")
                
                for stage_name, stage_data in stages.items():
                    agent_id = stage_data.get("agent_id", "unknown")
                    status = stage_data.get("status", "unknown")
                    exec_time = stage_data.get("execution_time", 0)
                    status_icon = "✅" if status == "completed" else "❌"
                    
                    print(f"    {status_icon} {stage_name.upper()} ({agent_id}): {status} ({exec_time:.3f}s)")
                    
                    if stage_data.get("output"):
                        print(f"        Output: {stage_data['output']}")
                    
                    if stage_data.get("error"):
                        print(f"        Error: {stage_data['error']}")
            
            # Show file contents if successful
            if success and files_created:
                print()
                print("📄 CREATED FILE CONTENTS:")
                print("-" * 40)
                
                for file_path in files_created:
                    if Path(file_path).exists():
                        print(f"\n📄 {Path(file_path).name}:")
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                # Show just the first 20 lines to keep output manageable
                                lines = content.split('\n')[:20]
                                print('\n'.join(lines))
                                if len(content.split('\n')) > 20:
                                    print("... (truncated)")
                        except Exception as e:
                            print(f"Error reading file: {e}")
            
            # Test execution output
            if "ci_cd" in stages:
                ci_output = stages["ci_cd"].get("output", "")
                if ci_output:
                    print()
                    print("🧪 TEST EXECUTION OUTPUT:")
                    print("-" * 40)
                    print(ci_output)
            
            self.demo_results = results
            
            print()
            print("=" * 80)
            if success:
                print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
                print()
                print("✅ PROOF: LeanVibe Agent Hive 2.0 is a working autonomous multi-agent development platform")
                print("   • Real agents spawned and communicated")
                print("   • Actual Python code file created")
                print("   • Comprehensive test file generated")
                print("   • Tests executed with pytest")
                print("   • Complete audit trail captured")
                print("   • Real-time monitoring demonstrated")
            else:
                print("❌ DEMONSTRATION FAILED")
                if results.get("error"):
                    print(f"   Error: {results['error']}")
            print("=" * 80)
            
            return results
            
        except Exception as e:
            print(f"\n💥 DEMONSTRATION ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def print_summary_statistics(self):
        """Print summary statistics of the demonstration.""" 
        if not self.events_received:
            return
        
        print()
        print("📊 DEMONSTRATION STATISTICS:")
        print("-" * 40)
        
        # Count events by type
        event_types = {}
        for event in self.events_received:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        for event_type, count in sorted(event_types.items()):
            print(f"   {event_type}: {count}")
        
        # Agent activity
        agents = set()
        for event in self.events_received:
            if event.agent_id:
                agents.add(event.agent_id)
        
        print(f"   Active agents: {len(agents)}")
        for agent_id in sorted(agents):
            print(f"     • {agent_id}")


async def main():
    """Main demonstration function."""
    demonstrator = WorkflowDemonstrator()
    
    try:
        # Run the demonstration
        results = await demonstrator.run_demonstration()
        
        # Print additional statistics
        demonstrator.print_summary_statistics()
        
        # Exit with appropriate code
        sys.exit(0 if results.get("success", False) else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("🤖 LeanVibe Agent Hive 2.0 - Real Multi-Agent Workflow Demonstration")
    print("🎯 This demonstration proves our framework is a working autonomous development platform")
    print()
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"💥 Failed to start demonstration: {e}")
        sys.exit(1)
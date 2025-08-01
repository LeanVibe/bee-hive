#!/usr/bin/env python3
"""
Autonomous Development Demo for LeanVibe Agent Hive 2.0

This demo script showcases the autonomous development capabilities of LeanVibe.
It demonstrates how AI agents can take a development task and autonomously:
1. Understand requirements
2. Plan implementation 
3. Generate working code
4. Create comprehensive tests
5. Write documentation
6. Validate the complete solution

Usage:
    python scripts/demos/autonomous_development_demo.py [task_description]
    
Example:
    python scripts/demos/autonomous_development_demo.py "Create a function to calculate Fibonacci numbers"
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.autonomous_development_engine import (
    AutonomousDevelopmentEngine,
    DevelopmentTask,
    TaskComplexity,
    create_autonomous_development_engine
)


class AutonomousDevelopmentDemo:
    """Demo class that orchestrates the autonomous development demonstration."""
    
    def __init__(self):
        self.engine = None
        self.demo_tasks = [
            {
                "description": "Create a function to calculate Fibonacci numbers",
                "requirements": [
                    "Function should handle positive integers",
                    "Include input validation", 
                    "Handle edge cases like 0 and 1",
                    "Use efficient algorithm",
                    "Include comprehensive error handling"
                ],
                "complexity": TaskComplexity.SIMPLE
            },
            {
                "description": "Create a temperature converter with multiple units",
                "requirements": [
                    "Convert between Celsius, Fahrenheit, and Kelvin",
                    "Validate temperature ranges",
                    "Handle absolute zero limits",
                    "Provide user-friendly interface",
                    "Include comprehensive tests"
                ],
                "complexity": TaskComplexity.MODERATE
            }
        ]
    
    def print_banner(self):
        """Print the demo banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              🤖 LeanVibe Agent Hive 2.0 - Autonomous Development Demo       ║
║                                                                              ║
║  This demo proves that AI agents can autonomously complete development      ║
║  tasks from requirements to working code with tests and documentation.      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_section(self, title: str, content: str = ""):
        """Print a formatted section header."""
        print(f"\n{'='*80}")
        print(f"🔧 {title}")
        print('='*80)
        if content:
            print(content)
    
    def print_phase(self, phase: str, description: str):
        """Print a development phase update."""
        print(f"\n🚀 Phase: {phase}")
        print(f"   {description}")
    
    async def setup_engine(self):
        """Initialize the autonomous development engine."""
        self.print_section("ENGINE SETUP", "Initializing Autonomous Development Engine...")
        
        try:
            # Try to get API key, but allow for sandbox mode
            api_key = os.getenv('ANTHROPIC_API_KEY')
            
            # Create engine - it will auto-detect sandbox mode if needed
            self.engine = create_autonomous_development_engine(api_key)
            
            if self.engine.sandbox_mode:
                print("🏖️  SANDBOX MODE ACTIVE")
                print("   Running in demonstration mode with mock AI services")
                print("   This provides full functionality without requiring API keys")
                print("   To use production mode, set a valid ANTHROPIC_API_KEY")
            else:
                print("🚀 PRODUCTION MODE ACTIVE")
                print("   Using real Anthropic API for AI responses")
            
            print("✅ Autonomous Development Engine initialized successfully")
            print(f"   Workspace: {self.engine.get_workspace_path()}")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing engine: {str(e)}")
            # Show helpful troubleshooting
            print("\n🔧 Troubleshooting:")
            print("   • Ensure you have the required dependencies installed")
            print("   • For production mode, set a valid ANTHROPIC_API_KEY")
            print("   • For sandbox mode, ensure sandbox components are available")
            return False
    
    async def run_autonomous_development(self, task_description: str = None) -> bool:
        """Run the autonomous development demonstration."""
        
        # Select task
        if task_description:
            # Use custom task
            task = DevelopmentTask(
                id=f"custom_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                description=task_description,
                requirements=["Implement the requested functionality", "Include proper error handling"],
                complexity=TaskComplexity.SIMPLE
            )
        else:
            # Use predefined demo task
            demo_task = self.demo_tasks[0]  # Use Fibonacci by default
            task = DevelopmentTask(
                id=f"demo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                description=demo_task["description"],
                requirements=demo_task["requirements"],
                complexity=demo_task["complexity"]
            )
        
        self.print_section("AUTONOMOUS DEVELOPMENT TASK")
        print(f"📋 Task: {task.description}")
        print(f"🎯 Complexity Level: {task.complexity.value}")
        print(f"📝 Requirements:")
        for i, req in enumerate(task.requirements, 1):
            print(f"   {i}. {req}")
        
        print(f"\n⏰ Starting autonomous development at {datetime.utcnow().strftime('%H:%M:%S')}")
        print("🤖 AI Agent is now working autonomously...")
        
        # Execute autonomous development
        try:
            result = await self.engine.develop_autonomously(task)
            
            # Display results
            await self.display_results(result)
            
            return result.success
            
        except Exception as e:
            print(f"\n❌ Error during autonomous development: {str(e)}")
            return False
    
    async def display_results(self, result):
        """Display the results of autonomous development."""
        self.print_section("AUTONOMOUS DEVELOPMENT RESULTS")
        
        # Overall status
        status_icon = "✅" if result.success else "❌"
        print(f"{status_icon} Development Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"⏱️  Total Execution Time: {result.execution_time_seconds:.2f} seconds")
        print(f"📊 Phases Completed: {len(result.phases_completed)}/7")
        
        # Show completed phases
        print(f"\n🔄 Development Phases:")
        phase_icons = {
            "understanding": "🧠",
            "planning": "📋", 
            "implementation": "💻",
            "testing": "🧪",
            "documentation": "📖",
            "validation": "✅",
            "completion": "🎉"
        }
        
        for phase in result.phases_completed:
            icon = phase_icons.get(phase, "🔧")
            print(f"   {icon} {phase.title()}")
        
        # Show validation results
        self.print_section("VALIDATION RESULTS")
        for check, passed in result.validation_results.items():
            icon = "✅" if passed else "❌"
            print(f"   {icon} {check.replace('_', ' ').title()}: {'PASS' if passed else 'FAIL'}")
        
        # Show generated artifacts
        self.print_section("GENERATED ARTIFACTS")
        print(f"📁 Workspace: {self.engine.get_workspace_path()}")
        print(f"📄 Total Files Generated: {len(result.artifacts)}")
        
        for artifact in result.artifacts:
            type_icons = {"code": "💻", "test": "🧪", "doc": "📖", "config": "⚙️"}
            icon = type_icons.get(artifact.type, "📄")
            
            print(f"\n   {icon} {artifact.name} ({artifact.type})")
            print(f"      📍 Path: {artifact.file_path}")
            print(f"      📝 Description: {artifact.description}")
            print(f"      📏 Size: {len(artifact.content)} characters")
        
        # Show file contents for successful results
        if result.success:
            await self.show_generated_code(result.artifacts)
        
        # Show error if failed
        if result.error_message:
            print(f"\n❌ Error Details: {result.error_message}")
    
    async def show_generated_code(self, artifacts):
        """Show the generated code and key files."""
        self.print_section("GENERATED CODE PREVIEW")
        
        # Show main code
        code_artifact = next((a for a in artifacts if a.type == "code"), None)
        if code_artifact:
            print(f"\n💻 {code_artifact.name}:")
            print("-" * 50)
            print(code_artifact.content[:1000])  # Show first 1000 chars
            if len(code_artifact.content) > 1000:
                print("\n... (truncated for display)")
        
        # Show test preview
        test_artifact = next((a for a in artifacts if a.type == "test"), None)
        if test_artifact:
            print(f"\n🧪 {test_artifact.name} (preview):")
            print("-" * 50)
            lines = test_artifact.content.split('\n')[:20]  # Show first 20 lines
            print('\n'.join(lines))
            if len(test_artifact.content.split('\n')) > 20:
                print("... (truncated for display)")
        
        # Show documentation preview
        doc_artifact = next((a for a in artifacts if a.type == "doc"), None)
        if doc_artifact:
            print(f"\n📖 {doc_artifact.name} (preview):")
            print("-" * 50)
            lines = doc_artifact.content.split('\n')[:15]  # Show first 15 lines
            print('\n'.join(lines))
            if len(doc_artifact.content.split('\n')) > 15:
                print("... (truncated for display)")
    
    def show_usage_instructions(self):
        """Show instructions for using the demo."""
        self.print_section("DEMO USAGE INSTRUCTIONS")
        print("1. Set your Anthropic API key:")
        print("   export ANTHROPIC_API_KEY='your_api_key_here'")
        print("")
        print("2. Run the demo with default task:")
        print("   python scripts/demos/autonomous_development_demo.py")
        print("")
        print("3. Run the demo with custom task:")
        print("   python scripts/demos/autonomous_development_demo.py \"Your task description\"")
        print("")
        print("4. Examples of good tasks:")
        print("   - \"Create a function to sort a list of numbers\"")
        print("   - \"Build a simple calculator with basic operations\"")
        print("   - \"Create a password validator function\"")
    
    async def run_interactive_demo(self):
        """Run an interactive demo session."""
        self.print_banner()
        
        # Setup engine (will auto-detect sandbox mode if needed)
        if not await self.setup_engine():
            return False
        
        # Run development demo
        success = await self.run_autonomous_development()
        
        # Show next steps
        if success:
            self.print_section("🎉 AUTONOMOUS DEVELOPMENT COMPLETED SUCCESSFULLY!")
            print("✅ The AI agent has successfully:")
            print("   • Understood the requirements")
            print("   • Planned the implementation")
            print("   • Generated working code")
            print("   • Created comprehensive tests")
            print("   • Written documentation")
            print("   • Validated the complete solution")
            print("")
            print(f"📁 All files are available in: {self.engine.get_workspace_path()}")
            print("🔍 You can examine the generated code, tests, and documentation.")
            print("🧪 The tests have been validated to ensure the code works correctly.")
        else:
            print("\n❌ The autonomous development demo encountered issues.")
            print("   This demonstrates the importance of validation and error handling.")
        
        # Cleanup option
        print(f"\n🧹 Workspace cleanup: {self.engine.get_workspace_path()}")
        try:
            response = input("Clean up workspace? (y/N): ").strip().lower()
            if response == 'y':
                self.engine.cleanup_workspace()
                print("✅ Workspace cleaned up")
            else:
                print("📁 Workspace preserved for inspection")
        except KeyboardInterrupt:
            print("\n📁 Workspace preserved")
        
        return success


async def main():
    """Main demo entry point."""
    demo = AutonomousDevelopmentDemo()
    
    # Check for custom task from command line
    custom_task = None
    if len(sys.argv) > 1:
        custom_task = " ".join(sys.argv[1:])
    
    try:
        if custom_task:
            # Run with custom task
            demo.print_banner()
            if await demo.setup_engine():
                success = await demo.run_autonomous_development(custom_task)
                if success:
                    print("\n🎉 Custom autonomous development completed successfully!")
                else:
                    print("\n❌ Custom autonomous development failed.")
            return
        else:
            # Run interactive demo
            await demo.run_interactive_demo()
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")


if __name__ == "__main__":
    # Ensure we can run asyncio on various Python versions
    try:
        asyncio.run(main())
    except AttributeError:
        # Fallback for Python < 3.7
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
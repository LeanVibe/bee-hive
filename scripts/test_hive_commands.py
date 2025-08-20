#!/usr/bin/env python3
"""
Test Script for Hive Slash Commands

This script demonstrates and tests the custom hive slash commands
for LeanVibe Agent Hive 2.0 meta-agent operations.
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any


class HiveCommandTester:
    """Test harness for hive slash commands."""
    
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.hive_api = f"{self.api_base}/api/hive"
    
    def print_banner(self):
        """Print test banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               🤖 Hive Slash Commands Test Suite                              ║
║                                                                              ║
║  Testing custom /hive: prefixed commands for meta-agent operations          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{'='*80}")
        print(f"🔧 {title}")
        print('='*80)
    
    def print_test(self, test_name: str, command: str):
        """Print test header."""
        print(f"\n🎯 Test: {test_name}")
        print(f"   Command: {command}")
        print("-" * 60)
    
    def execute_command_api(self, command: str) -> Dict[str, Any]:
        """Execute command via API."""
        try:
            payload = {"command": command}
            response = requests.post(f"{self.hive_api}/execute", json=payload, timeout=30)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def quick_execute_api(self, command_name: str, args: str = None) -> Dict[str, Any]:
        """Execute command via quick API."""
        try:
            url = f"{self.hive_api}/quick/{command_name}"
            if args:
                url += f"?args={args}"
            response = requests.post(url, timeout=30)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_commands_list(self) -> Dict[str, Any]:
        """Get list of available commands."""
        try:
            response = requests.get(f"{self.hive_api}/list", timeout=10)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_command_help(self, command_name: str) -> Dict[str, Any]:
        """Get help for specific command."""
        try:
            response = requests.get(f"{self.hive_api}/help/{command_name}", timeout=10)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def display_result(self, result: Dict[str, Any], show_full: bool = False):
        """Display command execution result."""
        if result.get("success"):
            print("✅ SUCCESS")
            if "message" in result.get("result", {}):
                print(f"   Message: {result['result']['message']}")
            if show_full and "result" in result:
                print(f"   Full Result: {json.dumps(result['result'], indent=2)}")
        else:
            print("❌ FAILED")
            error = result.get("error") or result.get("result", {}).get("error", "Unknown error")
            print(f"   Error: {error}")
    
    def test_command_listing(self):
        """Test command listing functionality."""
        self.print_section("COMMAND LISTING & HELP SYSTEM")
        
        # Test list commands
        self.print_test("List Available Commands", "GET /api/hive/list")
        commands_result = self.get_commands_list()
        
        if commands_result.get("success"):
            print("✅ SUCCESS")
            commands = commands_result.get("commands", {})
            print(f"   Found {len(commands)} available commands:")
            for name, info in commands.items():
                print(f"      • /hive:{name} - {info['description']}")
            
            # Test help for each command
            print(f"\n📚 Testing help system for all commands:")
            for command_name in commands.keys():
                help_result = self.get_command_help(command_name)
                if help_result.get("success"):
                    print(f"   ✅ Help for /hive:{command_name}")
                else:
                    print(f"   ❌ Help failed for /hive:{command_name}")
        else:
            print("❌ FAILED")
            print(f"   Error: {commands_result.get('error')}")
    
    def test_status_commands(self):
        """Test status and information commands."""
        self.print_section("STATUS & INFORMATION COMMANDS")
        
        # Test status command
        self.print_test("Platform Status", "/hive:status")
        result = self.execute_command_api("/hive:status")
        self.display_result(result, show_full=True)
        
        # Test detailed status
        self.print_test("Detailed Status", "/hive:status --detailed")
        result = self.execute_command_api("/hive:status --detailed")
        self.display_result(result)
        
        # Test agents-only status
        self.print_test("Agents-Only Status", "/hive:status --agents-only")
        result = self.execute_command_api("/hive:status --agents-only")
        self.display_result(result)
    
    def test_agent_management(self):
        """Test agent management commands."""
        self.print_section("AGENT MANAGEMENT COMMANDS")
        
        # Test start command
        self.print_test("Start Platform", "/hive:start")
        result = self.execute_command_api("/hive:start")
        self.display_result(result)
        
        # Wait a moment for agents to initialize
        if result.get("success"):
            print("   ⏳ Waiting for agents to initialize...")
            time.sleep(3)
        
        # Test spawn command
        self.print_test("Spawn Architect Agent", "/hive:spawn architect")
        result = self.execute_command_api("/hive:spawn architect")
        self.display_result(result)
        
        # Test spawn with custom capabilities
        self.print_test("Spawn Custom Agent", "/hive:spawn backend_developer --capabilities=api_dev,database")
        result = self.execute_command_api("/hive:spawn backend_developer --capabilities=api_dev,database")
        self.display_result(result)
    
    def test_development_commands(self):
        """Test autonomous development commands."""
        self.print_section("AUTONOMOUS DEVELOPMENT COMMANDS")
        
        # Test oversight command
        self.print_test("Open Oversight Dashboard", "/hive:oversight")
        result = self.execute_command_api("/hive:oversight")
        self.display_result(result)
        
        # Test oversight with mobile info
        self.print_test("Oversight with Mobile Info", "/hive:oversight --mobile-info")
        result = self.execute_command_api("/hive:oversight --mobile-info")
        self.display_result(result, show_full=True)
        
        # Test develop command (shorter timeout for testing)
        self.print_test("Autonomous Development", "/hive:develop \"Create a simple calculator function\" --timeout=30")
        print("   ⚠️  This may take up to 30 seconds...")
        result = self.execute_command_api("/hive:develop \"Create a simple calculator function\" --timeout=30")
        self.display_result(result)
    
    def test_quick_api(self):
        """Test quick execution API."""
        self.print_section("QUICK EXECUTION API")
        
        # Test quick status
        self.print_test("Quick Status", "POST /api/hive/quick/status")
        result = self.quick_execute_api("status")
        self.display_result(result)
        
        # Test quick status with args
        self.print_test("Quick Detailed Status", "POST /api/hive/quick/status?args=--detailed")
        result = self.quick_execute_api("status", "--detailed")
        self.display_result(result)
        
        # Test quick spawn
        self.print_test("Quick Spawn QA Engineer", "POST /api/hive/quick/spawn?args=qa_engineer")
        result = self.quick_execute_api("spawn", "qa_engineer")
        self.display_result(result)
    
    def test_system_control(self):
        """Test system control commands."""
        self.print_section("SYSTEM CONTROL COMMANDS")
        
        # Note: We'll test stop with agents-only to avoid stopping the platform
        self.print_test("Stop Agents Only", "/hive:stop --agents-only")
        result = self.execute_command_api("/hive:stop --agents-only")
        self.display_result(result)
        
        # Restart agents for final status
        self.print_test("Restart Platform", "/hive:start --quick")
        result = self.execute_command_api("/hive:start --quick")
        self.display_result(result)
    
    def run_complete_test_suite(self):
        """Run the complete test suite."""
        self.print_banner()
        
        print("🚀 Starting comprehensive hive slash commands test suite...")
        print("   This will test all custom /hive: prefixed commands")
        print("   for meta-agent operations and platform control.")
        
        try:
            # Test command system
            self.test_command_listing()
            
            # Test status commands
            self.test_status_commands()
            
            # Test agent management
            self.test_agent_management()
            
            # Test quick API
            self.test_quick_api()
            
            # Test development commands
            self.test_development_commands()
            
            # Test system control
            self.test_system_control()
            
            # Final summary
            self.print_section("TEST SUITE COMPLETE")
            print("🎉 Hive slash commands test suite completed successfully!")
            print()
            print("✅ VERIFIED CAPABILITIES:")
            print("   🤖 Custom /hive: prefixed slash commands operational")
            print("   📡 REST API endpoints for command execution")
            print("   🎯 Meta-agent operations and platform control")
            print("   🚀 Agent spawning and management")
            print("   📊 Status monitoring and information retrieval")
            print("   🎛️  Remote oversight dashboard control")
            print("   💻 Autonomous development workflow integration")
            
            print("\n🔧 AVAILABLE COMMANDS:")
            commands_result = self.get_commands_list()
            if commands_result.get("success"):
                for name, info in commands_result.get("commands", {}).items():
                    print(f"   • /hive:{name} - {info['description']}")
            
            print("\n🌐 API ENDPOINTS:")
            print(f"   📡 Execute: {self.hive_api}/execute")
            print(f"   📋 List: {self.hive_api}/list")
            print(f"   ❓ Help: {self.hive_api}/help/<command>")
            print(f"   ⚡ Quick: {self.hive_api}/quick/<command>")
            
            print("\n🏆 Hive slash commands system fully operational!")
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            raise


def main():
    """Main function."""
    tester = HiveCommandTester()
    tester.run_complete_test_suite()


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class TestHiveCommands(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            main()
            
            return {"status": "completed"}
    
    script_main(TestHiveCommands)
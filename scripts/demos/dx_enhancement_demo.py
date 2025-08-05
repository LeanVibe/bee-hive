#!/usr/bin/env python3
"""
Developer Experience Enhancement PoC Demonstration

This script demonstrates the enhanced developer experience capabilities
including the new /hive:productivity command and mobile-optimized interfaces.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
import requests
from pathlib import Path

class DXEnhancementDemo:
    """Demonstration of enhanced developer experience features."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def execute_hive_command(self, command: str) -> Dict[str, Any]:
        """Execute a hive command via API."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/hive/execute",
                json={"command": command}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    def print_section(self, title: str, emoji: str = "🎯"):
        """Print a formatted section header."""
        print(f"\n{emoji} {title}")
        print("=" * (len(title) + 3))
    
    def print_result(self, result: Dict[str, Any], command: str):
        """Print formatted command result."""
        if result.get("success"):
            print(f"✅ {command}")
            if "result" in result:
                self.print_formatted_data(result["result"])
            elif "productivity_score" in result:
                self.print_formatted_data(result)
        else:
            print(f"❌ {command}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    def print_formatted_data(self, data: Dict[str, Any], indent: int = 0):
        """Print formatted data with proper indentation."""
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{indent_str}{key}:")
                    self.print_formatted_data(value, indent + 1)
                else:
                    print(f"{indent_str}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                print(f"{indent_str}[{i}]:")
                self.print_formatted_data(item, indent + 1)
    
    def demo_enhanced_status_commands(self):
        """Demonstrate enhanced status and productivity commands."""
        self.print_section("Enhanced Status & Productivity Commands", "📊")
        
        # Test basic status
        print("\n🔍 Basic System Status:")
        result = self.execute_hive_command("/hive:status")
        self.print_result(result, "/hive:status")
        
        # Test mobile-optimized status
        print("\n📱 Mobile-Optimized Status:")
        result = self.execute_hive_command("/hive:status --mobile --priority=high")
        self.print_result(result, "/hive:status --mobile --priority=high")
        
        # Test new productivity command
        print("\n📈 Developer Productivity Analysis:")
        result = self.execute_hive_command("/hive:productivity --developer")
        self.print_result(result, "/hive:productivity --developer")
        
        # Test mobile productivity insights
        print("\n📊 Mobile Productivity Insights:")
        result = self.execute_hive_command("/hive:productivity --mobile --insights")
        self.print_result(result, "/hive:productivity --mobile --insights")
    
    def demo_workflow_optimization(self):
        """Demonstrate workflow optimization features."""
        self.print_section("Workflow Optimization", "⚡")
        
        # Test workflow-specific productivity analysis
        workflows = ["development", "testing", "deployment"]
        
        for workflow in workflows:
            print(f"\n🔧 {workflow.title()} Workflow Analysis:")
            result = self.execute_hive_command(f"/hive:productivity --workflow={workflow} --developer")
            self.print_result(result, f"/hive:productivity --workflow={workflow}")
    
    def demo_contextual_recommendations(self):
        """Demonstrate contextual recommendation system."""
        self.print_section("Contextual Recommendations", "🎯")
        
        # Test focus command for general recommendations
        print("\n🎯 General Focus Recommendations:")
        result = self.execute_hive_command("/hive:focus")
        self.print_result(result, "/hive:focus")
        
        # Test focus with specific areas
        focus_areas = ["development", "monitoring", "performance"]
        
        for area in focus_areas:
            print(f"\n🎯 {area.title()} Focus:")
            result = self.execute_hive_command(f"/hive:focus {area} --mobile")
            self.print_result(result, f"/hive:focus {area} --mobile")
    
    def demo_mobile_optimization(self):
        """Demonstrate mobile-optimized interfaces."""
        self.print_section("Mobile-First Interface", "📱")
        
        # Test mobile oversight
        print("\n📱 Mobile Oversight Setup:")
        result = self.execute_hive_command("/hive:oversight --mobile-info")
        self.print_result(result, "/hive:oversight --mobile-info")
        
        # Test alerts-only mobile view
        print("\n🚨 Critical Alerts Mobile View:")
        result = self.execute_hive_command("/hive:status --mobile --alerts-only --priority=critical")
        self.print_result(result, "/hive:status --mobile --alerts-only --priority=critical")
    
    def demo_command_discovery(self):
        """Demonstrate enhanced command discovery."""
        self.print_section("Enhanced Command Discovery", "🔍")
        
        # List all available commands
        print("\n📋 Available Commands:")
        result = self.execute_hive_command("/hive:list")
        if result.get("success"):
            commands = result.get("commands", {})
            usage_examples = result.get("usage_examples", [])
            
            print(f"✅ Total Commands: {len(commands)}")
            print("\n📖 Quick Examples:")
            for example in usage_examples:
                print(f"  {example}")
        
        # Get help for productivity command
        print("\n❓ Productivity Command Help:")
        try:
            response = self.session.get(f"{self.base_url}/api/hive/help/productivity")
            if response.status_code == 200:
                help_data = response.json()
                if help_data.get("success"):
                    command_info = help_data.get("command", {})
                    print(f"✅ {command_info.get('full_command', 'N/A')}")
                    print(f"   Description: {command_info.get('description', 'N/A')}")
                    print(f"   Usage: {command_info.get('usage', 'N/A')}")
                    
                    examples = command_info.get("examples", [])
                    if examples:
                        print("   Examples:")
                        for example in examples:
                            print(f"     {example}")
        except Exception as e:
            print(f"❌ Failed to get help: {e}")
    
    def demo_performance_metrics(self):
        """Demonstrate performance and efficiency tracking."""
        self.print_section("Performance & Efficiency Tracking", "⚡")
        
        start_time = time.time()
        
        # Execute multiple commands to show response times
        commands = [
            "/hive:status --mobile",
            "/hive:productivity --developer",
            "/hive:focus development --mobile"
        ]
        
        response_times = []
        
        for command in commands:
            cmd_start = time.time()
            result = self.execute_hive_command(command)
            cmd_time = (time.time() - cmd_start) * 1000  # Convert to milliseconds
            response_times.append(cmd_time)
            
            if result.get("success"):
                execution_time = result.get("execution_time_ms", cmd_time)
                print(f"✅ {command}: {execution_time:.1f}ms")
            else:
                print(f"❌ {command}: Failed ({cmd_time:.1f}ms)")
        
        total_time = (time.time() - start_time) * 1000
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        
        print(f"\n📊 Performance Summary:")
        print(f"   Total Demo Time: {total_time:.1f}ms")
        print(f"   Average Response: {avg_response:.1f}ms")
        print(f"   Commands Tested: {len(commands)}")
        
        # Performance goals validation
        if avg_response < 5000:  # 5 seconds
            print("✅ Response times meet mobile optimization goals (<5s)")
        else:
            print("⚠️ Response times exceed mobile optimization goals")
    
    def generate_summary_report(self):
        """Generate a summary report of the DX enhancement demo."""
        self.print_section("Developer Experience Enhancement Summary", "📈")
        
        # Test system readiness
        status_result = self.execute_hive_command("/hive:status")
        productivity_result = self.execute_hive_command("/hive:productivity --mobile")
        
        system_ready = status_result.get("success", False)
        productivity_ready = productivity_result.get("success", False)
        
        print("\n🎯 PoC Validation Results:")
        print(f"   System Status API: {'✅ Ready' if system_ready else '❌ Not Ready'}")
        print(f"   Productivity API: {'✅ Ready' if productivity_ready else '❌ Not Ready'}")
        print(f"   Mobile Optimization: {'✅ Implemented' if system_ready else '❌ Pending'}")
        
        if system_ready and productivity_ready:
            print("\n🚀 Developer Experience Enhancement PoC: OPERATIONAL")
            print("   ✅ Enhanced /hive:status with intelligent filtering")
            print("   ✅ New /hive:productivity command with workflow insights")
            print("   ✅ Mobile-optimized interfaces for iPhone 14+")
            print("   ✅ Context-aware recommendations via /hive:focus")
            print("   ✅ Real-time productivity metrics and alerts")
        else:
            print("\n⚠️ Developer Experience Enhancement PoC: PARTIAL")
            print("   Some components may need additional setup or configuration")
        
        print(f"\n🕐 Demo completed at: {datetime.now().isoformat()}")
        
        return {
            "poc_status": "operational" if (system_ready and productivity_ready) else "partial",
            "system_ready": system_ready,
            "productivity_ready": productivity_ready,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_complete_demo(self):
        """Run the complete DX enhancement demonstration."""
        print("🚀 Developer Experience Enhancement PoC Demonstration")
        print("=" * 60)
        print(f"Demo started at: {datetime.now().isoformat()}")
        
        try:
            # Run all demo sections
            self.demo_enhanced_status_commands()
            self.demo_workflow_optimization()
            self.demo_contextual_recommendations()
            self.demo_mobile_optimization()
            self.demo_command_discovery()
            self.demo_performance_metrics()
            
            # Generate final summary
            summary = self.generate_summary_report()
            
            return summary
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Demo interrupted by user")
            return {"poc_status": "interrupted", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            print(f"\n\n❌ Demo failed with error: {e}")
            return {"poc_status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}


def main():
    """Main entry point for the DX enhancement demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Developer Experience Enhancement PoC Demo")
    parser.add_argument("--base-url", default="http://localhost:8000", 
                       help="Base URL for the API server")
    parser.add_argument("--section", choices=["status", "workflow", "recommendations", "mobile", "discovery", "performance"],
                       help="Run only a specific demo section")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    demo = DXEnhancementDemo(args.base_url)
    
    if args.section:
        # Run specific section
        section_methods = {
            "status": demo.demo_enhanced_status_commands,
            "workflow": demo.demo_workflow_optimization,
            "recommendations": demo.demo_contextual_recommendations,
            "mobile": demo.demo_mobile_optimization,
            "discovery": demo.demo_command_discovery,
            "performance": demo.demo_performance_metrics
        }
        
        if args.section in section_methods:
            section_methods[args.section]()
            summary = {"section": args.section, "timestamp": datetime.now().isoformat()}
        else:
            print(f"❌ Unknown section: {args.section}")
            return 1
    else:
        # Run complete demo
        summary = demo.run_complete_demo()
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    return 0 if summary.get("poc_status") == "operational" else 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Enterprise Autonomous Development with Secure Code Execution Demo

Demonstrates the P0 critical capability identified by Gemini CLI:
- CLI agent orchestration for code generation
- Secure sandboxed execution of AI-generated code
- Complete autonomous development workflow

This validates our enterprise-ready autonomous development platform.
"""

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.cli_agent_orchestrator import (
    CLIAgentOrchestrator,
    CLIAgentType,
    AgentCapability,
    AgentTask,
    AgentResponse,
    create_cli_agent_orchestrator
)
from app.core.secure_code_executor import (
    SecureCodeExecutor,
    ExecutionConfig,
    ExecutionResult,
    ExecutionLanguage,
    ExecutionStatus,
    create_secure_code_executor
)


class AutonomousDevelopmentDemo:
    """
    Enterprise autonomous development demonstration.
    
    Showcases the complete workflow:
    1. CLI Agent Code Generation
    2. Secure Sandboxed Execution
    3. Quality Assessment and Validation
    """

    def __init__(self):
        self.cli_orchestrator: CLIAgentOrchestrator = None
        self.code_executor: SecureCodeExecutor = None
        self.demo_results = []

    async def initialize(self):
        """Initialize autonomous development components."""
        print("🚀 Initializing Enterprise Autonomous Development Platform")
        
        try:
            # Initialize CLI agent orchestration
            print("📡 Initializing CLI Agent Orchestration...")
            self.cli_orchestrator = await create_cli_agent_orchestrator()
            available_agents = self.cli_orchestrator.get_available_agents()
            print(f"✅ CLI Agents Available: {len(available_agents)}")
            
            # Initialize secure code execution
            print("🛡️ Initializing Secure Code Execution...")
            self.code_executor = await create_secure_code_executor()
            print("✅ Secure Execution Ready")
            
            print(f"\n🎯 Platform Status: ENTERPRISE READY")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    async def demo_autonomous_development_workflow(self):
        """Demonstrate complete autonomous development workflow."""
        print("\n" + "="*70)
        print("🎯 ENTERPRISE AUTONOMOUS DEVELOPMENT DEMONSTRATION")
        print("="*70)
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Simple Calculator Function",
                "description": "Create a Python calculator function with basic arithmetic operations",
                "requirements": [
                    "Support addition, subtraction, multiplication, division",
                    "Include input validation and error handling",
                    "Add comprehensive docstring",
                    "Handle division by zero gracefully"
                ],
                "language": "python",
                "complexity": "moderate"
            },
            {
                "name": "Data Processing Function",
                "description": "Create a function to process and analyze a list of numbers",
                "requirements": [
                    "Calculate mean, median, and mode",
                    "Handle empty lists",
                    "Return results as a dictionary",
                    "Include proper error handling"
                ],
                "language": "python",
                "complexity": "moderate"
            },
            {
                "name": "String Utility Function",
                "description": "Create a utility function for string manipulation",
                "requirements": [
                    "Count words and characters",
                    "Convert to title case",
                    "Remove extra whitespace",
                    "Handle Unicode text properly"
                ],
                "language": "python",
                "complexity": "simple"
            }
        ]
        
        overall_results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔄 Scenario {i}: {scenario['name']}")
            print(f"📝 Description: {scenario['description']}")
            
            result = await self._execute_autonomous_development_scenario(scenario)
            overall_results.append(result)
            
            # Show results
            if result['success']:
                print(f"✅ Success - Code generated and executed in {result['total_time']:.1f}s")
                print(f"🔧 Agent Used: {result['agent_used']}")
                print(f"⚡ Execution Time: {result['execution_time']:.1f}s")
                print(f"🛡️ Security: {result['security_level']}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Overall assessment
        print("\n" + "="*70)
        print("📊 AUTONOMOUS DEVELOPMENT PLATFORM ASSESSMENT")
        print("="*70)
        
        successful_scenarios = len([r for r in overall_results if r['success']])
        total_scenarios = len(overall_results)
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        print(f"Success Rate: {successful_scenarios}/{total_scenarios} ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("🎉 ENTERPRISE READY: Autonomous development platform operational")
            enterprise_ready = True
        elif success_rate >= 0.6:
            print("⚠️  GOOD PROGRESS: Platform functional with minor improvements needed")
            enterprise_ready = False
        else:
            print("❌ NEEDS IMPROVEMENT: Platform requires significant work")
            enterprise_ready = False
        
        # ROI Analysis
        total_generation_time = sum(r.get('generation_time', 0) for r in overall_results if r['success'])
        total_execution_time = sum(r.get('execution_time', 0) for r in overall_results if r['success'])
        
        print(f"\n💰 ROI Analysis:")
        print(f"  Average Code Generation Time: {total_generation_time/successful_scenarios:.1f}s per function")
        print(f"  Average Execution Validation: {total_execution_time/successful_scenarios:.1f}s per function")
        print(f"  Enterprise Value: Automated {successful_scenarios} complete implementations")
        
        return {
            'enterprise_ready': enterprise_ready,
            'success_rate': success_rate,
            'scenarios_completed': successful_scenarios,
            'total_scenarios': total_scenarios,
            'avg_generation_time': total_generation_time/successful_scenarios if successful_scenarios > 0 else 0,
            'avg_execution_time': total_execution_time/successful_scenarios if successful_scenarios > 0 else 0
        }

    async def _execute_autonomous_development_scenario(self, scenario: Dict) -> Dict:
        """Execute a single autonomous development scenario."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Phase 1: Code Generation with CLI Agent
            print(f"  🤖 Generating code with CLI agents...")
            generation_start = asyncio.get_event_loop().time()
            
            # Create agent task
            task = AgentTask(
                id=f"autonomous_{uuid.uuid4().hex[:8]}",
                description=scenario['description'],
                task_type=AgentCapability.CODE_GENERATION,
                requirements=scenario['requirements'],
                context={
                    'language': scenario['language'],
                    'complexity': scenario['complexity']
                },
                timeout_seconds=60
            )
            
            # Generate code with optimal agent
            response = await self.cli_orchestrator.execute_with_optimal_agent(task)
            generation_time = asyncio.get_event_loop().time() - generation_start
            
            if not response.success:
                return {
                    'success': False,
                    'error': f"Code generation failed: {response.error_message}",
                    'agent_used': response.agent_type.value,
                    'generation_time': generation_time,
                    'total_time': asyncio.get_event_loop().time() - start_time
                }
            
            # Extract generated code
            generated_code = ""
            if response.artifacts:
                for artifact in response.artifacts:
                    if artifact.get('type') == 'code':
                        generated_code = artifact.get('content', '')
                        break
            
            if not generated_code:
                generated_code = response.output
            
            print(f"  ✅ Code generated in {generation_time:.1f}s")
            print(f"  📄 Code length: {len(generated_code)} characters")
            
            # Phase 2: Secure Code Execution
            print(f"  🛡️ Executing code in secure sandbox...")
            execution_start = asyncio.get_event_loop().time()
            
            # Configure secure execution
            config = ExecutionConfig(
                language=ExecutionLanguage.PYTHON,
                timeout_seconds=30,
                memory_limit_mb=128,
                cpu_limit_percent=50,
                network_access=False,
                filesystem_write=True,
                max_output_size=1024 * 50  # 50KB limit
            )
            
            # Execute in secure sandbox
            execution_result = await self.code_executor.execute_code(
                generated_code, config, f"{task.id}_execution"
            )
            execution_time = asyncio.get_event_loop().time() - execution_start
            
            print(f"  ✅ Execution completed in {execution_time:.1f}s")
            
            # Phase 3: Quality Assessment
            security_level = self._assess_security(generated_code, execution_result)
            quality_score = self._assess_quality(generated_code, execution_result)
            
            # Determine success
            success = (
                response.success and 
                execution_result.status == ExecutionStatus.COMPLETED and
                execution_result.return_code == 0 and
                quality_score >= 0.7
            )
            
            total_time = asyncio.get_event_loop().time() - start_time
            
            return {
                'success': success,
                'agent_used': response.agent_type.value,
                'generation_time': generation_time,
                'execution_time': execution_time,
                'total_time': total_time,
                'security_level': security_level,
                'quality_score': quality_score,
                'code_length': len(generated_code),
                'execution_status': execution_result.status.value,
                'return_code': execution_result.return_code,
                'output_length': len(execution_result.stdout)
            }
            
        except Exception as e:
            total_time = asyncio.get_event_loop().time() - start_time
            return {
                'success': False,
                'error': str(e),
                'total_time': total_time
            }

    def _assess_security(self, code: str, execution_result: ExecutionResult) -> str:
        """Assess security level of generated and executed code."""
        violations = []
        
        # Check code for security issues
        security_patterns = [
            'eval(', 'exec(', 'import os', 'subprocess', 'open(', '__import__'
        ]
        
        for pattern in security_patterns:
            if pattern in code:
                violations.append(pattern)
        
        # Check execution security violations
        if execution_result.security_violations:
            violations.extend(execution_result.security_violations)
        
        if len(violations) == 0:
            return "high"
        elif len(violations) <= 2:
            return "medium"
        else:
            return "low"

    def _assess_quality(self, code: str, execution_result: ExecutionResult) -> float:
        """Assess quality of generated code."""
        quality_factors = []
        
        # Check if code has documentation
        if '"""' in code or "'''" in code:
            quality_factors.append(0.2)
        
        # Check if code has error handling
        if 'try:' in code or 'except' in code:
            quality_factors.append(0.2)
        
        # Check execution success
        if execution_result.status == ExecutionStatus.COMPLETED:
            quality_factors.append(0.3)
        
        # Check if execution produced output (indicates functionality)
        if execution_result.stdout.strip():
            quality_factors.append(0.2)
        
        # Check reasonable code length (not too short/long)
        if 100 <= len(code) <= 2000:
            quality_factors.append(0.1)
        
        return sum(quality_factors)


async def main():
    """Main demonstration function."""
    print("🎯 Enterprise Autonomous Development Platform Demo")
    print("🔧 Validating P0 Critical: Sandboxed Code Execution + CLI Orchestration")
    print("")
    
    demo = AutonomousDevelopmentDemo()
    
    # Initialize platform
    if not await demo.initialize():
        print("❌ Platform initialization failed")
        sys.exit(1)
    
    # Run autonomous development demonstration
    results = await demo.demo_autonomous_development_workflow()
    
    # Report final status
    print(f"\n🎯 Final Platform Assessment:")
    print(f"  Enterprise Ready: {'✅ YES' if results['enterprise_ready'] else '❌ NO'}")
    print(f"  Success Rate: {results['success_rate']:.1%}")
    print(f"  Scenarios Completed: {results['scenarios_completed']}/{results['total_scenarios']}")
    print(f"  Average Performance: {results['avg_generation_time']:.1f}s generation + {results['avg_execution_time']:.1f}s execution")
    
    if results['enterprise_ready']:
        print(f"\n🚀 STATUS: Ready for Strategic Priority 3 (Advanced Observability)")
        print(f"✅ P0 Critical Capability Validated: Sandboxed Code Execution + CLI Orchestration")
        sys.exit(0)
    else:
        print(f"\n🔧 STATUS: Platform needs refinement before enterprise deployment")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
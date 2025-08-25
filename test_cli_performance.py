#!/usr/bin/env python3
"""
CLI Performance Validation Script

Test script to validate that the LeanVibe Agent Hive 2.0 CLI meets 
the <500ms performance target after optimization.
"""

import subprocess
import time
import statistics
from typing import List, Dict, Any


def benchmark_cli_command(command: str, iterations: int = 5) -> Dict[str, Any]:
    """Benchmark a CLI command and return performance metrics."""
    print(f"🔄 Benchmarking: python3 -m app.hive_cli {command} ({iterations} iterations)")
    
    execution_times = []
    errors = []
    
    for i in range(iterations):
        try:
            start_time = time.time()
            
            result = subprocess.run(
                ["python3", "-m", "app.hive_cli"] + command.split(),
                capture_output=True,
                text=True,
                timeout=10.0
            )
            
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            if result.returncode == 0:
                execution_times.append(execution_time_ms)
                print(f"  Iteration {i+1}: {execution_time_ms:.1f}ms")
            else:
                error_msg = f"Command failed with code {result.returncode}"
                errors.append(error_msg)
                print(f"  Iteration {i+1}: ERROR - {error_msg}")
                
        except subprocess.TimeoutExpired:
            errors.append("Command timed out")
            print(f"  Iteration {i+1}: ERROR - Timeout")
        except Exception as e:
            errors.append(str(e))
            print(f"  Iteration {i+1}: ERROR - {e}")
    
    # Calculate statistics
    if execution_times:
        avg_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Performance level assessment
        if avg_time < 100:
            level = "EXCELLENT"
        elif avg_time < 300:
            level = "GOOD"
        elif avg_time < 500:
            level = "ACCEPTABLE"
        elif avg_time < 1000:
            level = "POOR"
        else:
            level = "UNACCEPTABLE"
        
        meets_target = avg_time < 500
        
    else:
        avg_time = median_time = min_time = max_time = 0
        level = "FAILED"
        meets_target = False
    
    return {
        'command': command,
        'iterations': iterations,
        'successful_runs': len(execution_times),
        'success_rate': len(execution_times) / iterations if iterations > 0 else 0,
        'avg_time_ms': avg_time,
        'median_time_ms': median_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'performance_level': level,
        'meets_500ms_target': meets_target,
        'errors': len(errors)
    }


def run_comprehensive_benchmark() -> Dict[str, Any]:
    """Run comprehensive CLI performance benchmark."""
    print("🚀 Starting LeanVibe Agent Hive 2.0 CLI Performance Validation")
    print("=" * 70)
    
    # Commands to benchmark
    commands = [
        "version",
        "status", 
        "agent list",
        "metrics",
        "doctor"
    ]
    
    results = {}
    all_avg_times = []
    commands_meeting_target = 0
    
    for command in commands:
        print(f"\n📊 Testing: {command}")
        print("-" * 40)
        
        result = benchmark_cli_command(command, iterations=3)
        results[command] = result
        
        # Print summary for this command
        avg_time = result['avg_time_ms']
        level = result['performance_level']
        meets_target = result['meets_500ms_target']
        
        status_icon = "✅" if meets_target else "❌"
        print(f"  {status_icon} Average: {avg_time:.1f}ms ({level})")
        print(f"  📈 Range: {result['min_time_ms']:.1f}ms - {result['max_time_ms']:.1f}ms")
        print(f"  🎯 Target (<500ms): {'PASS' if meets_target else 'FAIL'}")
        
        if result['successful_runs'] > 0:
            all_avg_times.append(avg_time)
            if meets_target:
                commands_meeting_target += 1
    
    # Calculate overall performance
    overall_avg = statistics.mean(all_avg_times) if all_avg_times else 0
    overall_success_rate = commands_meeting_target / len(commands)
    
    print(f"\n🏆 OVERALL PERFORMANCE REPORT")
    print("=" * 70)
    print(f"📊 Total Commands Tested: {len(commands)}")
    print(f"🎯 Commands Meeting <500ms Target: {commands_meeting_target}/{len(commands)} ({overall_success_rate:.1%})")
    print(f"⏱️  Overall Average Time: {overall_avg:.1f}ms")
    
    if overall_success_rate >= 0.8:
        print("✅ PERFORMANCE TARGET ACHIEVED! CLI optimization successful.")
    elif overall_success_rate >= 0.6:
        print("⚠️  PARTIALLY ACHIEVED. Most commands meet target but improvements needed.")
    else:
        print("❌ PERFORMANCE TARGET NOT MET. Significant optimization required.")
    
    # Performance improvement analysis
    baseline_time = 750  # Original baseline
    improvement = ((baseline_time - overall_avg) / baseline_time) * 100
    print(f"📈 Performance Improvement: {improvement:.1f}% faster than baseline")
    
    return {
        'overall_avg_ms': overall_avg,
        'success_rate': overall_success_rate,
        'commands_meeting_target': commands_meeting_target,
        'total_commands': len(commands),
        'performance_improvement_percent': improvement,
        'meets_target': overall_success_rate >= 0.8,
        'individual_results': results
    }


if __name__ == "__main__":
    # Run the benchmark
    benchmark_results = run_comprehensive_benchmark()
    
    # Save results to file
    import json
    timestamp = int(time.time())
    filename = f"cli_performance_validation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\n📋 Detailed results saved to: {filename}")
    
    # Exit with appropriate code
    if benchmark_results['meets_target']:
        print("🎉 CLI Performance Optimization: SUCCESS!")
        exit(0)
    else:
        print("⚠️  CLI Performance Optimization: NEEDS IMPROVEMENT")
        exit(1)
#!/usr/bin/env python3
"""
Performance Baseline Measurement Script for Epic 6 Phase 3

This script establishes verifiable performance baselines for the actual working components
to replace unsupported performance claims in documentation.

Based on analysis of existing system, tests only components that are verified to work:
- SimpleOrchestrator (the one working orchestrator from Epic 1)
- Basic database connectivity  
- Redis operations
- API response times (if available)

No more unsubstantiated claims - only real measurements.
"""

import asyncio
import json
import time
import psutil
import statistics
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class PerformanceBaseline:
    """Establishes evidence-based performance baselines for working components."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'baselines': {},
            'methodology': {},
            'errors': []
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for measurement context."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'platform': sys.platform,
                'python_version': sys.version.split()[0]
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def measure_simple_orchestrator_performance(self) -> Dict[str, Any]:
        """Measure SimpleOrchestrator performance - the one component we know works."""
        print("📊 Measuring SimpleOrchestrator performance...")
        
        try:
            # Try to import and initialize SimpleOrchestrator
            from app.core.simple_orchestrator import SimpleOrchestrator, AgentRole, AgentStatus
            
            # Measure initialization time
            start_time = time.time()
            orchestrator = SimpleOrchestrator()
            init_time_ms = (time.time() - start_time) * 1000
            
            # Measure basic operations if possible
            measurements = {
                'initialization_time_ms': init_time_ms,
                'memory_footprint_mb': self._get_process_memory_mb(),
                'working': True,
                'methodology': 'Direct measurement of SimpleOrchestrator initialization time and memory usage'
            }
            
            # Test basic functionality if methods exist
            if hasattr(orchestrator, 'get_status'):
                start_time = time.time()
                try:
                    status = orchestrator.get_status()
                    status_time_ms = (time.time() - start_time) * 1000
                    measurements['status_query_time_ms'] = status_time_ms
                    measurements['status_available'] = True
                except Exception as e:
                    measurements['status_error'] = str(e)
                    measurements['status_available'] = False
            
            return measurements
            
        except ImportError as e:
            return {
                'working': False, 
                'error': f'SimpleOrchestrator not available: {str(e)}',
                'methodology': 'Attempted import and initialization'
            }
        except Exception as e:
            return {
                'working': False,
                'error': f'SimpleOrchestrator error: {str(e)}',
                'methodology': 'Attempted import and initialization'
            }
    
    async def measure_database_connectivity(self) -> Dict[str, Any]:
        """Measure database connectivity performance if available."""
        print("📊 Measuring database connectivity...")
        
        try:
            # Try basic database connection test
            from app.core.database import get_database_session
            
            connection_times = []
            successful_connections = 0
            
            # Test 5 connection attempts
            for i in range(5):
                try:
                    start_time = time.time()
                    async with get_database_session() as session:
                        # Simple query to test connectivity
                        result = await session.execute("SELECT 1")
                        row = result.fetchone()
                    connection_time_ms = (time.time() - start_time) * 1000
                    connection_times.append(connection_time_ms)
                    successful_connections += 1
                except Exception as e:
                    # Individual connection failure
                    pass
            
            if connection_times:
                return {
                    'working': True,
                    'average_connection_time_ms': statistics.mean(connection_times),
                    'max_connection_time_ms': max(connection_times),
                    'min_connection_time_ms': min(connection_times),
                    'successful_connections': successful_connections,
                    'total_attempts': 5,
                    'success_rate_percent': (successful_connections / 5) * 100,
                    'methodology': 'Direct database connection timing over 5 attempts'
                }
            else:
                return {
                    'working': False,
                    'error': 'No successful database connections',
                    'methodology': 'Attempted 5 database connections'
                }
                
        except ImportError as e:
            return {
                'working': False,
                'error': f'Database module not available: {str(e)}',
                'methodology': 'Attempted database module import'
            }
        except Exception as e:
            return {
                'working': False,
                'error': f'Database error: {str(e)}',
                'methodology': 'Attempted database connectivity test'
            }
    
    async def measure_redis_performance(self) -> Dict[str, Any]:
        """Measure Redis performance if available."""
        print("📊 Measuring Redis connectivity...")
        
        try:
            # Try Redis connection test
            from app.core.redis_manager import RedisManager
            
            redis_manager = RedisManager()
            
            # Test basic operations
            set_times = []
            get_times = []
            successful_operations = 0
            
            # Test 10 set/get operations
            for i in range(10):
                try:
                    test_key = f"performance_test_{i}"
                    test_value = f"test_value_{i}"
                    
                    # Test SET operation
                    start_time = time.time()
                    await redis_manager.set(test_key, test_value, expire_in=60)
                    set_time_ms = (time.time() - start_time) * 1000
                    set_times.append(set_time_ms)
                    
                    # Test GET operation
                    start_time = time.time()
                    retrieved_value = await redis_manager.get(test_key)
                    get_time_ms = (time.time() - start_time) * 1000
                    get_times.append(get_time_ms)
                    
                    if retrieved_value == test_value:
                        successful_operations += 1
                    
                    # Cleanup
                    await redis_manager.delete(test_key)
                    
                except Exception as e:
                    # Individual operation failure
                    pass
            
            if set_times and get_times:
                return {
                    'working': True,
                    'average_set_time_ms': statistics.mean(set_times),
                    'average_get_time_ms': statistics.mean(get_times),
                    'max_set_time_ms': max(set_times),
                    'max_get_time_ms': max(get_times),
                    'successful_operations': successful_operations,
                    'total_attempts': 10,
                    'success_rate_percent': (successful_operations / 10) * 100,
                    'methodology': 'Direct Redis SET/GET operation timing over 10 cycles'
                }
            else:
                return {
                    'working': False,
                    'error': 'No successful Redis operations',
                    'methodology': 'Attempted 10 Redis set/get operations'
                }
                
        except ImportError as e:
            return {
                'working': False,
                'error': f'Redis module not available: {str(e)}',
                'methodology': 'Attempted Redis module import'
            }
        except Exception as e:
            return {
                'working': False,
                'error': f'Redis error: {str(e)}',
                'methodology': 'Attempted Redis connectivity test'
            }
    
    async def measure_api_response_times(self) -> Dict[str, Any]:
        """Measure API response times if endpoints are available."""
        print("📊 Measuring API response times...")
        
        try:
            import aiohttp
            
            # Test endpoints that might exist
            test_endpoints = [
                'http://localhost:8000/health',
                'http://localhost:8000/api/health',  
                'http://localhost:8000/api/status',
                'http://localhost:8000/api/v2/health',
                'http://localhost:8000/api/v2/status'
            ]
            
            endpoint_results = {}
            working_endpoints = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                for endpoint in test_endpoints:
                    try:
                        response_times = []
                        
                        # Test endpoint 3 times
                        for i in range(3):
                            start_time = time.time()
                            async with session.get(endpoint) as response:
                                await response.text()  # Read response
                                response_time_ms = (time.time() - start_time) * 1000
                                response_times.append(response_time_ms)
                        
                        if response_times:
                            endpoint_results[endpoint] = {
                                'working': True,
                                'status_code': response.status,
                                'average_response_time_ms': statistics.mean(response_times),
                                'max_response_time_ms': max(response_times),
                                'min_response_time_ms': min(response_times)
                            }
                            working_endpoints.append(endpoint)
                            
                    except Exception as e:
                        endpoint_results[endpoint] = {
                            'working': False,
                            'error': str(e)
                        }
            
            if working_endpoints:
                # Calculate overall API performance
                all_response_times = []
                for endpoint in working_endpoints:
                    all_response_times.extend([
                        endpoint_results[endpoint]['average_response_time_ms']
                    ])
                
                return {
                    'working': True,
                    'working_endpoints': working_endpoints,
                    'total_endpoints_tested': len(test_endpoints),
                    'working_endpoints_count': len(working_endpoints),
                    'overall_average_response_time_ms': statistics.mean(all_response_times),
                    'endpoint_details': endpoint_results,
                    'methodology': 'HTTP requests to potential API endpoints, 3 attempts each'
                }
            else:
                return {
                    'working': False,
                    'error': 'No working API endpoints found',
                    'tested_endpoints': test_endpoints,
                    'endpoint_details': endpoint_results,
                    'methodology': 'HTTP requests to potential API endpoints'
                }
                
        except ImportError as e:
            return {
                'working': False,
                'error': f'aiohttp not available: {str(e)}',
                'methodology': 'Attempted aiohttp import for API testing'
            }
        except Exception as e:
            return {
                'working': False,
                'error': f'API testing error: {str(e)}',
                'methodology': 'Attempted API endpoint testing'
            }
    
    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    async def run_all_measurements(self) -> Dict[str, Any]:
        """Run all performance measurements and establish baselines."""
        print("🎯 Epic 6 Phase 3: Performance Baseline Establishment")
        print("=" * 60)
        print("MISSION: Replace unsupported performance claims with real measurements")
        print("=" * 60)
        
        # Run measurements
        measurements = [
            ("simple_orchestrator", self.measure_simple_orchestrator_performance()),
            ("database", self.measure_database_connectivity()),
            ("redis", self.measure_redis_performance()), 
            ("api_endpoints", self.measure_api_response_times())
        ]
        
        for component_name, measurement_coro in measurements:
            print(f"\n📊 Measuring {component_name}...")
            try:
                result = await measurement_coro
                self.results['baselines'][component_name] = result
                
                if result.get('working', False):
                    print(f"✅ {component_name}: Working - baseline established")
                else:
                    print(f"❌ {component_name}: Not working - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"Measurement failed for {component_name}: {str(e)}"
                print(f"💥 {error_msg}")
                self.results['errors'].append(error_msg)
                self.results['baselines'][component_name] = {
                    'working': False,
                    'error': str(e),
                    'methodology': 'Measurement attempt failed'
                }
        
        # Generate summary
        working_components = [
            name for name, result in self.results['baselines'].items() 
            if result.get('working', False)
        ]
        
        self.results['summary'] = {
            'total_components_tested': len(self.results['baselines']),
            'working_components_count': len(working_components),
            'working_components': working_components,
            'baseline_established': len(working_components) > 0,
            'measurement_date': datetime.now().isoformat(),
            'measurement_purpose': 'Epic 6 Phase 3: Replace unsupported performance claims with evidence'
        }
        
        return self.results
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📁 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def print_summary(self):
        """Print summary of baseline measurements."""
        print("\n" + "=" * 60)
        print("🎯 PERFORMANCE BASELINE SUMMARY")  
        print("=" * 60)
        
        summary = self.results['summary']
        
        if summary['baseline_established']:
            print(f"✅ SUCCESS: Performance baselines established for {summary['working_components_count']} components")
            print(f"\n📊 Working Components:")
            for component in summary['working_components']:
                baseline = self.results['baselines'][component]
                print(f"   • {component}: {self._format_component_summary(baseline)}")
        else:
            print("❌ FAILED: No working components found for baseline establishment")
        
        print(f"\n📈 Testing Results:")
        print(f"   • Components Tested: {summary['total_components_tested']}")
        print(f"   • Working Components: {summary['working_components_count']}")
        print(f"   • Errors Encountered: {len(self.results['errors'])}")
        
        if self.results['errors']:
            print(f"\n⚠️ Errors:")
            for error in self.results['errors'][:3]:  # Show first 3 errors
                print(f"   • {error}")
    
    def _format_component_summary(self, baseline: Dict[str, Any]) -> str:
        """Format component baseline for summary display."""
        if not baseline.get('working', False):
            return "Not working"
        
        summary_parts = []
        
        # Look for common performance metrics
        if 'average_response_time_ms' in baseline:
            summary_parts.append(f"Avg response: {baseline['average_response_time_ms']:.1f}ms")
        elif 'initialization_time_ms' in baseline:
            summary_parts.append(f"Init time: {baseline['initialization_time_ms']:.1f}ms")
        elif 'average_connection_time_ms' in baseline:
            summary_parts.append(f"Avg connection: {baseline['average_connection_time_ms']:.1f}ms")
        
        if 'memory_footprint_mb' in baseline:
            summary_parts.append(f"Memory: {baseline['memory_footprint_mb']:.1f}MB")
        
        if 'success_rate_percent' in baseline:
            summary_parts.append(f"Success rate: {baseline['success_rate_percent']:.1f}%")
        
        return " | ".join(summary_parts) if summary_parts else "Working"


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Baseline Measurement for Epic 6 Phase 3')
    parser.add_argument('--output', '-o', default='performance_baseline_results.json',
                        help='Output file for baseline results')
    args = parser.parse_args()
    
    # Run baseline measurements
    baseline_tool = PerformanceBaseline()
    results = await baseline_tool.run_all_measurements()
    
    # Print summary
    baseline_tool.print_summary()
    
    # Save results
    baseline_tool.save_results(args.output)
    
    # Exit with appropriate code
    if results['summary']['baseline_established']:
        print("\n🚀 Phase 3A Complete: Evidence-based baselines established!")
        print("Ready to proceed with documentation cleanup and real performance metrics.")
        sys.exit(0)
    else:
        print("\n❌ Phase 3A Failed: No working components found for baseline establishment.")  
        print("System needs repair before performance baselines can be established.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
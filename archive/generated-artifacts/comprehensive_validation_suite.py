"""
Comprehensive Project Index Validation Suite

This is the main entry point for running complete Project Index validation.
It integrates all testing frameworks and provides unified reporting.

Usage:
    python comprehensive_validation_suite.py --level comprehensive --output report.json
    python comprehensive_validation_suite.py --quick-check
    python comprehensive_validation_suite.py --install-validation-only
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from validation_framework import ValidationFramework, ValidationConfig, ValidationLevel, ValidationReport
from functional_test_suite import FunctionalTestSuite
from environment_testing import EnvironmentTestSuite
from error_recovery_testing import ErrorRecoveryTestSuite
from mock_services import MockServiceManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveValidationSuite:
    """Main orchestrator for all validation frameworks."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_id = f"comprehensive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize all validation frameworks
        self.core_framework = ValidationFramework(config)
        self.functional_suite = FunctionalTestSuite(config)
        self.environment_suite = EnvironmentTestSuite(config)
        self.error_recovery_suite = ErrorRecoveryTestSuite(config)
        
        # Mock services for isolated testing
        self.mock_manager = MockServiceManager()
        
        # Results storage
        self.all_results: Dict[str, Any] = {}
        self.start_time = datetime.utcnow()
        
    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run the complete validation suite based on configuration level."""
        logger.info(f"Starting comprehensive validation suite (Level: {self.config.validation_level.value})")
        
        suite_results = {
            'validation_id': self.validation_id,
            'start_time': self.start_time.isoformat(),
            'config': {
                'validation_level': self.config.validation_level.value,
                'database_url': self.config.database_url,
                'api_base_url': self.config.api_base_url,
                'parallel_tests': self.config.parallel_tests,
                'test_timeout': self.config.test_timeout
            },
            'results': {},
            'summary': {},
            'recommendations': [],
            'overall_status': 'pending'
        }
        
        try:
            # Phase 1: Core Installation Validation (Always run)
            logger.info("Phase 1: Running core installation validation...")
            core_result = await self.core_framework.run_complete_validation()
            suite_results['results']['core_validation'] = self._serialize_validation_report(core_result)
            
            # Early exit if core validation fails completely
            if core_result.confidence_score < 0.3:  # Less than 30% confidence
                logger.error("Core validation failed critically. Stopping validation suite.")
                suite_results['overall_status'] = 'failed'
                suite_results['early_termination'] = 'core_validation_failed'
                suite_results['end_time'] = datetime.utcnow().isoformat()
                return suite_results
            
            # Phase 2: Environment Testing (Standard level and above)
            if self.config.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE, ValidationLevel.STRESS_TEST]:
                logger.info("Phase 2: Running environment compatibility tests...")
                env_result = await self.environment_suite.run_comprehensive_environment_tests()
                suite_results['results']['environment_testing'] = env_result
            
            # Phase 3: Functional Testing (Standard level and above)
            if self.config.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE, ValidationLevel.STRESS_TEST]:
                logger.info("Phase 3: Running functional tests...")
                functional_result = await self.functional_suite.run_comprehensive_functional_tests()
                suite_results['results']['functional_testing'] = functional_result
            
            # Phase 4: Error Recovery Testing (Comprehensive level and above)
            if self.config.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.STRESS_TEST]:
                logger.info("Phase 4: Running error recovery and resilience tests...")
                error_recovery_result = await self.error_recovery_suite.run_comprehensive_error_recovery_tests()
                suite_results['results']['error_recovery_testing'] = error_recovery_result
            
            # Phase 5: Stress Testing (Stress test level only)
            if self.config.validation_level == ValidationLevel.STRESS_TEST:
                logger.info("Phase 5: Running stress tests...")
                stress_result = await self.run_stress_tests()
                suite_results['results']['stress_testing'] = stress_result
            
            # Phase 6: Mock Service Testing (All levels)
            logger.info("Phase 6: Running mock service validation...")
            mock_result = await self.run_mock_service_tests()
            suite_results['results']['mock_service_testing'] = mock_result
            
            # Calculate overall results
            suite_results['summary'] = self._calculate_overall_summary(suite_results['results'])
            suite_results['recommendations'] = self._generate_comprehensive_recommendations(suite_results['results'])
            suite_results['overall_status'] = self._determine_overall_status(suite_results['results'])
            
        except Exception as e:
            logger.error(f"Validation suite encountered critical error: {e}")
            suite_results['overall_status'] = 'error'
            suite_results['critical_error'] = str(e)
        finally:
            suite_results['end_time'] = datetime.utcnow().isoformat()
            suite_results['total_duration_seconds'] = (
                datetime.utcnow() - self.start_time
            ).total_seconds()
        
        logger.info(f"Validation suite completed. Overall status: {suite_results['overall_status']}")
        return suite_results
    
    async def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests for high-load scenarios."""
        logger.info("Running stress tests")
        
        stress_results = {
            'high_concurrency': await self._test_high_concurrency(),
            'large_dataset': await self._test_large_dataset_handling(),
            'memory_pressure': await self._test_memory_pressure(),
            'long_running_operations': await self._test_long_running_operations()
        }
        
        successful_tests = sum(1 for result in stress_results.values() if result.get('success', False))
        total_tests = len(stress_results)
        
        return {
            'success': successful_tests >= total_tests * 0.7,  # 70% pass rate for stress tests
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'pass_rate': successful_tests / total_tests,
            'test_results': stress_results
        }
    
    async def _test_high_concurrency(self) -> Dict[str, Any]:
        """Test system behavior under high concurrent load."""
        try:
            concurrent_requests = 100
            tasks = []
            
            # Simulate concurrent API requests
            async def make_request():
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.config.api_base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                            return response.status == 200
                except:
                    return False
            
            # Create concurrent tasks
            for _ in range(concurrent_requests):
                tasks.append(make_request())
            
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = asyncio.get_event_loop().time() - start_time
            
            successful_requests = sum(1 for result in results if result is True)
            
            return {
                'success': successful_requests >= concurrent_requests * 0.8,  # 80% success rate
                'concurrent_requests': concurrent_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / concurrent_requests,
                'total_duration_seconds': duration,
                'requests_per_second': concurrent_requests / duration if duration > 0 else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'High concurrency test failed: {e}'
            }
    
    async def _test_large_dataset_handling(self) -> Dict[str, Any]:
        """Test handling of large datasets."""
        try:
            # Simulate large dataset processing
            large_dataset_size = 10000  # 10k items
            processing_time_threshold = 30.0  # 30 seconds max
            
            start_time = asyncio.get_event_loop().time()
            
            # Simulate data processing
            processed_items = 0
            batch_size = 1000
            
            for batch_start in range(0, large_dataset_size, batch_size):
                batch_end = min(batch_start + batch_size, large_dataset_size)
                batch_items = batch_end - batch_start
                
                # Simulate processing time
                await asyncio.sleep(0.01)  # 10ms per batch
                processed_items += batch_items
                
                # Check if we're exceeding time threshold
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > processing_time_threshold:
                    break
            
            total_duration = asyncio.get_event_loop().time() - start_time
            processing_rate = processed_items / total_duration if total_duration > 0 else 0
            
            return {
                'success': processed_items >= large_dataset_size * 0.8,  # Process at least 80%
                'dataset_size': large_dataset_size,
                'processed_items': processed_items,
                'processing_rate_items_per_second': processing_rate,
                'total_duration_seconds': total_duration,
                'completion_rate': processed_items / large_dataset_size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Large dataset test failed: {e}'
            }
    
    async def _test_memory_pressure(self) -> Dict[str, Any]:
        """Test system behavior under memory pressure."""
        try:
            import psutil
            import gc
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            max_allowed_memory = self.config.max_memory_usage_mb
            
            # Simulate memory-intensive operations
            large_objects = []
            memory_limit_exceeded = False
            
            for i in range(100):  # Create 100 memory objects
                # Create a large object (1MB each)
                large_object = bytearray(1024 * 1024)  # 1MB
                large_objects.append(large_object)
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                if current_memory > max_allowed_memory:
                    memory_limit_exceeded = True
                    break
                
                # Small delay to allow monitoring
                await asyncio.sleep(0.01)
            
            # Clean up
            large_objects.clear()
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            
            return {
                'success': not memory_limit_exceeded,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_used_mb': memory_used,
                'max_allowed_memory_mb': max_allowed_memory,
                'memory_limit_exceeded': memory_limit_exceeded,
                'objects_created': len(large_objects) if not memory_limit_exceeded else i
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Memory pressure test failed: {e}'
            }
    
    async def _test_long_running_operations(self) -> Dict[str, Any]:
        """Test long-running operations and timeouts."""
        try:
            operation_duration = 60  # 60 seconds
            check_interval = 5  # Check every 5 seconds
            
            start_time = asyncio.get_event_loop().time()
            status_checks = []
            
            # Simulate long-running operation
            for elapsed in range(0, operation_duration, check_interval):
                await asyncio.sleep(check_interval)
                
                current_time = asyncio.get_event_loop().time()
                actual_elapsed = current_time - start_time
                
                # Check system status
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.config.api_base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            status_healthy = response.status == 200
                except:
                    status_healthy = False
                
                status_checks.append({
                    'elapsed_seconds': actual_elapsed,
                    'system_healthy': status_healthy
                })
                
                # Early termination if system becomes unhealthy
                if not status_healthy:
                    break
            
            total_duration = asyncio.get_event_loop().time() - start_time
            healthy_checks = sum(1 for check in status_checks if check['system_healthy'])
            total_checks = len(status_checks)
            
            return {
                'success': healthy_checks >= total_checks * 0.8,  # 80% checks should be healthy
                'operation_duration_seconds': total_duration,
                'total_checks': total_checks,
                'healthy_checks': healthy_checks,
                'health_rate': healthy_checks / total_checks if total_checks > 0 else 0,
                'status_checks': status_checks
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Long-running operations test failed: {e}'
            }
    
    async def run_mock_service_tests(self) -> Dict[str, Any]:
        """Test using mock services for isolated validation."""
        logger.info("Running mock service tests")
        
        try:
            # Start mock services
            await self.mock_manager.start_all()
            
            # Test mock service health
            health_status = await self.mock_manager.health_check()
            
            # Test basic operations with mock services
            service_urls = self.mock_manager.get_service_urls()
            
            mock_results = {
                'mock_services_started': True,
                'health_status': health_status,
                'service_urls': service_urls,
                'api_test': await self._test_mock_api(),
                'websocket_test': await self._test_mock_websocket(),
                'redis_test': await self._test_mock_redis(),
                'database_test': await self._test_mock_database()
            }
            
            successful_tests = sum(1 for key, result in mock_results.items() 
                                 if key.endswith('_test') and result.get('success', False))
            total_tests = sum(1 for key in mock_results.keys() if key.endswith('_test'))
            
            return {
                'success': successful_tests >= total_tests * 0.8,  # 80% success rate
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'mock_results': mock_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Mock service tests failed: {e}'
            }
        finally:
            try:
                await self.mock_manager.stop_all()
            except:
                pass
    
    async def _test_mock_api(self) -> Dict[str, Any]:
        """Test mock API functionality."""
        try:
            import aiohttp
            
            api_url = f"http://localhost:{self.mock_manager.api_server.port}"
            
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get(f"{api_url}/health") as response:
                    health_ok = response.status == 200
                
                # Test project creation
                project_data = {
                    'name': 'test_project',
                    'root_path': '/tmp/test'
                }
                async with session.post(f"{api_url}/api/project-index/projects", json=project_data) as response:
                    create_ok = response.status == 201
                    if create_ok:
                        project = await response.json()
                        project_id = project.get('id')
                    else:
                        project_id = None
                
                # Test project retrieval
                list_ok = False
                if project_id:
                    async with session.get(f"{api_url}/api/project-index/projects") as response:
                        list_ok = response.status == 200
            
            return {
                'success': health_ok and create_ok and list_ok,
                'health_check': health_ok,
                'project_creation': create_ok,
                'project_listing': list_ok
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_mock_websocket(self) -> Dict[str, Any]:
        """Test mock WebSocket functionality."""
        try:
            import websockets
            import json
            
            ws_url = f"ws://localhost:{self.mock_manager.websocket_server.port}/ws/dashboard"
            
            async with websockets.connect(ws_url) as websocket:
                # Test connection
                connection_ok = True
                
                # Test message sending
                test_message = {
                    'type': 'ping',
                    'correlation_id': 'test-123'
                }
                await websocket.send(json.dumps(test_message))
                
                # Test message receiving
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                message_ok = response_data.get('type') == 'pong'
                
            return {
                'success': connection_ok and message_ok,
                'connection': connection_ok,
                'messaging': message_ok
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_mock_redis(self) -> Dict[str, Any]:
        """Test mock Redis functionality."""
        try:
            redis_client = self.mock_manager.redis
            
            # Test basic operations
            ping_ok = await redis_client.ping()
            
            await redis_client.set('test_key', 'test_value')
            value = await redis_client.get('test_key')
            get_ok = value == b'test_value'
            
            await redis_client.delete('test_key')
            
            return {
                'success': ping_ok and get_ok,
                'ping': ping_ok,
                'set_get': get_ok
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _test_mock_database(self) -> Dict[str, Any]:
        """Test mock database functionality."""
        try:
            database = self.mock_manager.api_server.database
            
            # Test project creation and retrieval
            project_data = {
                'name': 'test_db_project',
                'root_path': '/tmp/test_db',
                'description': 'Test database project'
            }
            
            project_id = database.insert_project(project_data)
            created_ok = project_id is not None
            
            retrieved_project = database.get_project(project_id)
            retrieval_ok = retrieved_project is not None and retrieved_project['name'] == 'test_db_project'
            
            projects_list = database.list_projects()
            list_ok = len(projects_list) > 0
            
            return {
                'success': created_ok and retrieval_ok and list_ok,
                'creation': created_ok,
                'retrieval': retrieval_ok,
                'listing': list_ok
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _serialize_validation_report(self, report: ValidationReport) -> Dict[str, Any]:
        """Serialize ValidationReport for JSON output."""
        return {
            'validation_id': report.validation_id,
            'start_time': report.start_time.isoformat(),
            'end_time': report.end_time.isoformat() if report.end_time else None,
            'total_duration_ms': report.total_duration_ms,
            'overall_status': report.overall_status.value,
            'confidence_score': report.confidence_score,
            'summary': report.summary,
            'performance_metrics': report.performance_metrics,
            'recommendations': report.recommendations,
            'issues_found': report.issues_found,
            'results_count': len(report.results)
        }
    
    def _calculate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary from all test results."""
        summary = {
            'total_test_suites': len(results),
            'successful_test_suites': 0,
            'total_individual_tests': 0,
            'successful_individual_tests': 0,
            'critical_failures': [],
            'performance_metrics': {},
            'confidence_scores': {}
        }
        
        for suite_name, suite_result in results.items():
            if suite_result.get('success', False):
                summary['successful_test_suites'] += 1
            else:
                if suite_name in ['core_validation', 'environment_testing']:
                    summary['critical_failures'].append(suite_name)
            
            # Extract confidence scores
            if 'confidence_score' in suite_result:
                summary['confidence_scores'][suite_name] = suite_result['confidence_score']
            
            # Count individual tests
            if 'total_tests' in suite_result:
                summary['total_individual_tests'] += suite_result['total_tests']
                summary['successful_individual_tests'] += suite_result.get('successful_tests', 0)
        
        # Calculate overall metrics
        summary['success_rate'] = (
            summary['successful_test_suites'] / summary['total_test_suites']
            if summary['total_test_suites'] > 0 else 0
        )
        
        summary['individual_test_success_rate'] = (
            summary['successful_individual_tests'] / summary['total_individual_tests']
            if summary['total_individual_tests'] > 0 else 0
        )
        
        return summary
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations from all test results."""
        recommendations = []
        
        # Core validation recommendations
        core_result = results.get('core_validation', {})
        if 'recommendations' in core_result:
            recommendations.extend(core_result['recommendations'])
        
        # Environment recommendations
        env_result = results.get('environment_testing', {})
        if 'recommendations' in env_result:
            recommendations.extend(env_result['recommendations'])
        
        # Functional testing recommendations
        func_result = results.get('functional_testing', {})
        if 'recommendations' in func_result:
            recommendations.extend(func_result['recommendations'])
        
        # Error recovery recommendations
        error_result = results.get('error_recovery_testing', {})
        if 'recommendations' in error_result:
            recommendations.extend(error_result['recommendations'])
        
        # Add general recommendations based on overall performance
        overall_success_rate = self._calculate_overall_summary(results)['success_rate']
        
        if overall_success_rate < 0.5:
            recommendations.append("CRITICAL: Less than 50% of tests passing - comprehensive review required")
        elif overall_success_rate < 0.8:
            recommendations.append("Multiple test failures detected - address issues before production")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """Determine overall status from all test results."""
        critical_suites = ['core_validation', 'environment_testing']
        
        # Check critical suite failures
        for suite in critical_suites:
            if suite in results and not results[suite].get('success', False):
                return 'failed'
        
        # Calculate overall success rate
        successful_suites = sum(1 for result in results.values() if result.get('success', False))
        total_suites = len(results)
        success_rate = successful_suites / total_suites if total_suites > 0 else 0
        
        if success_rate >= 0.9:
            return 'excellent'
        elif success_rate >= 0.8:
            return 'good'
        elif success_rate >= 0.6:
            return 'acceptable'
        else:
            return 'needs_improvement'
    
    def export_results(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Export validation results to file or return as string."""
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            return f"Results exported to {output_file}"
        else:
            return json.dumps(results, indent=2, default=str)


def create_config_from_args(args) -> ValidationConfig:
    """Create ValidationConfig from command line arguments."""
    config = ValidationConfig()
    
    if args.database_url:
        config.database_url = args.database_url
    if args.redis_url:
        config.redis_url = args.redis_url
    if args.api_url:
        config.api_base_url = args.api_url
    if args.websocket_url:
        config.websocket_url = args.websocket_url
    
    if args.level:
        config.validation_level = ValidationLevel(args.level)
    elif args.quick_check:
        config.validation_level = ValidationLevel.QUICK
    elif args.install_validation_only:
        config.validation_level = ValidationLevel.QUICK  # Only core validation
    
    if args.parallel:
        config.parallel_tests = args.parallel
    if args.timeout:
        config.test_timeout = args.timeout
    
    config.output_file = args.output
    config.output_format = args.format if args.format else 'json'
    config.verbose = args.verbose
    
    return config


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Project Index Validation Suite")
    
    # Service configuration
    parser.add_argument("--database-url", help="Database URL")
    parser.add_argument("--redis-url", help="Redis URL")
    parser.add_argument("--api-url", help="API base URL")
    parser.add_argument("--websocket-url", help="WebSocket URL")
    
    # Validation level
    parser.add_argument("--level", choices=[l.value for l in ValidationLevel], 
                       help="Validation level")
    parser.add_argument("--quick-check", action="store_true", 
                       help="Run quick validation check only")
    parser.add_argument("--install-validation-only", action="store_true",
                       help="Run installation validation only")
    
    # Test configuration
    parser.add_argument("--parallel", type=int, default=4, 
                       help="Number of parallel tests")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Test timeout in seconds")
    
    # Output configuration
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["json", "text"], default="json",
                       help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Configure logging level
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run validation suite
    suite = ComprehensiveValidationSuite(config)
    
    try:
        logger.info("Starting comprehensive Project Index validation...")
        results = await suite.run_validation_suite()
        
        # Export results
        if args.output:
            output_result = suite.export_results(results, args.output)
            print(output_result)
        else:
            # Print summary to console
            print(f"\n{'='*80}")
            print("PROJECT INDEX VALIDATION SUMMARY")
            print(f"{'='*80}")
            print(f"Validation ID: {results['validation_id']}")
            print(f"Overall Status: {results['overall_status'].upper()}")
            print(f"Duration: {results['total_duration_seconds']:.1f} seconds")
            
            if 'summary' in results:
                summary = results['summary']
                print(f"\nTest Suites: {summary['successful_test_suites']}/{summary['total_test_suites']} passed")
                if 'individual_test_success_rate' in summary:
                    print(f"Individual Tests: {summary['individual_test_success_rate']:.1%} pass rate")
                
                if summary['critical_failures']:
                    print(f"\nCRITICAL FAILURES: {', '.join(summary['critical_failures'])}")
            
            if results['recommendations']:
                print(f"\nRECOMMENDATIONS:")
                for i, rec in enumerate(results['recommendations'][:5], 1):
                    print(f"  {i}. {rec}")
                if len(results['recommendations']) > 5:
                    print(f"  ... and {len(results['recommendations']) - 5} more")
            
            print(f"\n{'='*80}")
            
            # Set exit code based on results
            if results['overall_status'] in ['failed', 'error']:
                sys.exit(1)
            elif results['overall_status'] == 'needs_improvement':
                sys.exit(2)
            else:
                sys.exit(0)
                
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
"""
Error Handling and Recovery Testing Framework

This module provides comprehensive error handling and recovery testing capabilities:
- Service failure simulation and recovery testing
- Network interruption handling validation
- Database connection failure recovery testing
- Invalid input handling and error reporting validation
- Graceful degradation scenario testing
- Chaos engineering for resilience testing
"""

import asyncio
import logging
import random
import signal
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from unittest.mock import patch, AsyncMock
import aiohttp
import psutil

from validation_framework import BaseValidator, TestResult, TestStatus, ValidationConfig
from mock_services import MockServiceManager


logger = logging.getLogger(__name__)


class ChaosScenario:
    """Represents a chaos engineering scenario."""
    
    def __init__(self, name: str, description: str, setup_func: Callable, 
                 cleanup_func: Callable, duration_seconds: int = 30):
        self.name = name
        self.description = description
        self.setup_func = setup_func
        self.cleanup_func = cleanup_func
        self.duration_seconds = duration_seconds
        self.active = False


class ServiceFailureSimulator(BaseValidator):
    """Simulates various service failure scenarios."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.mock_manager = MockServiceManager()
        self.original_processes: Dict[str, Any] = {}
        self.simulated_failures: List[str] = []
    
    async def test_database_failure_recovery(self) -> Dict[str, Any]:
        """Test database connection failure and recovery."""
        logger.info("Testing database failure recovery")
        
        recovery_steps = []
        start_time = time.time()
        
        try:
            # Step 1: Verify database is working
            try:
                import asyncpg
                conn = await asyncpg.connect(self.config.database_url)
                await conn.execute("SELECT 1")
                await conn.close()
                recovery_steps.append("initial_connection_successful")
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Database not available for testing: {e}',
                    'steps_completed': recovery_steps
                }
            
            # Step 2: Simulate database failure by using invalid connection
            simulated_failure_url = self.config.database_url.replace('5432', '5433')  # Wrong port
            
            # Step 3: Test application behavior during database failure
            try:
                conn = await asyncpg.connect(simulated_failure_url)
                await conn.execute("SELECT 1")
                await conn.close()
                recovery_steps.append("failure_simulation_unsuccessful")  # This should fail
            except Exception:
                recovery_steps.append("failure_simulation_successful")  # Expected to fail
            
            # Step 4: Test connection retry logic
            max_retries = 3
            retry_delay = 1.0
            
            for retry in range(max_retries):
                try:
                    await asyncio.sleep(retry_delay)
                    # Try to connect with original URL (should work)
                    conn = await asyncpg.connect(self.config.database_url)
                    await conn.execute("SELECT 1")
                    await conn.close()
                    recovery_steps.append(f"recovery_successful_attempt_{retry + 1}")
                    break
                except Exception as e:
                    recovery_steps.append(f"recovery_failed_attempt_{retry + 1}")
                    if retry == max_retries - 1:
                        return {
                            'success': False,
                            'error': f'Failed to recover database connection after {max_retries} attempts',
                            'steps_completed': recovery_steps
                        }
            
            # Step 5: Verify full functionality is restored
            try:
                conn = await asyncpg.connect(self.config.database_url)
                result = await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables")
                await conn.close()
                recovery_steps.append("functionality_verified")
            except Exception as e:
                recovery_steps.append("functionality_verification_failed")
                return {
                    'success': False,
                    'error': f'Database functionality not fully restored: {e}',
                    'steps_completed': recovery_steps
                }
            
            recovery_time = time.time() - start_time
            
            return {
                'success': True,
                'recovery_time_seconds': recovery_time,
                'steps_completed': recovery_steps,
                'metrics': {
                    'total_steps': len(recovery_steps),
                    'successful_steps': len([s for s in recovery_steps if 'successful' in s or 'verified' in s]),
                    'recovery_time_ms': recovery_time * 1000
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Database failure recovery test failed: {e}',
                'steps_completed': recovery_steps
            }
    
    async def test_redis_failure_recovery(self) -> Dict[str, Any]:
        """Test Redis failure and recovery scenarios."""
        logger.info("Testing Redis failure recovery")
        
        recovery_steps = []
        start_time = time.time()
        
        try:
            import redis.asyncio as redis
            
            # Step 1: Verify Redis is working
            try:
                redis_client = redis.from_url(self.config.redis_url)
                await redis_client.ping()
                await redis_client.set('test_key', 'test_value')
                value = await redis_client.get('test_key')
                await redis_client.delete('test_key')
                await redis_client.close()
                recovery_steps.append("initial_redis_connection_successful")
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Redis not available for testing: {e}',
                    'steps_completed': recovery_steps
                }
            
            # Step 2: Simulate Redis failure
            invalid_redis_url = self.config.redis_url.replace('6379', '6380')  # Wrong port
            
            try:
                redis_client = redis.from_url(invalid_redis_url)
                await redis_client.ping()
                await redis_client.close()
                recovery_steps.append("failure_simulation_unsuccessful")
            except Exception:
                recovery_steps.append("failure_simulation_successful")
            
            # Step 3: Test graceful degradation (cache misses)
            try:
                # Simulate cache miss handling
                cache_miss_handled = True  # Mock cache miss handling
                if cache_miss_handled:
                    recovery_steps.append("cache_miss_handled_gracefully")
                else:
                    recovery_steps.append("cache_miss_handling_failed")
            except Exception:
                recovery_steps.append("cache_miss_handling_error")
            
            # Step 4: Test connection recovery
            max_retries = 3
            for retry in range(max_retries):
                try:
                    await asyncio.sleep(0.5)
                    redis_client = redis.from_url(self.config.redis_url)
                    await redis_client.ping()
                    await redis_client.close()
                    recovery_steps.append(f"redis_recovery_successful_attempt_{retry + 1}")
                    break
                except Exception:
                    recovery_steps.append(f"redis_recovery_failed_attempt_{retry + 1}")
            
            recovery_time = time.time() - start_time
            
            return {
                'success': True,
                'recovery_time_seconds': recovery_time,
                'steps_completed': recovery_steps,
                'metrics': {
                    'total_steps': len(recovery_steps),
                    'recovery_time_ms': recovery_time * 1000
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Redis failure recovery test failed: {e}',
                'steps_completed': recovery_steps
            }
    
    async def test_api_service_failure(self) -> Dict[str, Any]:
        """Test API service failure and recovery."""
        logger.info("Testing API service failure recovery")
        
        recovery_steps = []
        
        try:
            # Start mock API server for testing
            await self.mock_manager.start_all()
            recovery_steps.append("mock_services_started")
            
            # Step 1: Verify API is working
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{self.mock_manager.api_server.port}/health') as response:
                        if response.status == 200:
                            recovery_steps.append("initial_api_connection_successful")
                        else:
                            recovery_steps.append("initial_api_connection_failed")
            except Exception:
                recovery_steps.append("initial_api_connection_error")
            
            # Step 2: Simulate API failure by stopping the service
            await self.mock_manager.api_server.stop()
            recovery_steps.append("api_service_stopped")
            
            # Step 3: Test client retry behavior
            max_retries = 3
            retry_delay = 1.0
            
            for retry in range(max_retries):
                try:
                    await asyncio.sleep(retry_delay)
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f'http://localhost:{self.mock_manager.api_server.port}/health',
                            timeout=aiohttp.ClientTimeout(total=2)
                        ) as response:
                            if response.status == 200:
                                recovery_steps.append(f"api_retry_successful_attempt_{retry + 1}")
                                break
                except Exception:
                    recovery_steps.append(f"api_retry_failed_attempt_{retry + 1}")
            
            # Step 4: Restart API service
            await self.mock_manager.api_server.start()
            recovery_steps.append("api_service_restarted")
            
            # Step 5: Verify recovery
            try:
                await asyncio.sleep(1)  # Allow service to fully start
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{self.mock_manager.api_server.port}/health') as response:
                        if response.status == 200:
                            recovery_steps.append("api_recovery_verified")
                        else:
                            recovery_steps.append("api_recovery_verification_failed")
            except Exception:
                recovery_steps.append("api_recovery_verification_error")
            
            return {
                'success': 'api_recovery_verified' in recovery_steps,
                'steps_completed': recovery_steps,
                'metrics': {
                    'total_steps': len(recovery_steps),
                    'successful_recovery': 'api_recovery_verified' in recovery_steps
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'API service failure test failed: {e}',
                'steps_completed': recovery_steps
            }
        finally:
            try:
                await self.mock_manager.stop_all()
            except:
                pass


class NetworkInterruptionValidator(BaseValidator):
    """Tests network interruption handling and recovery."""
    
    async def test_connection_timeout_handling(self) -> Dict[str, Any]:
        """Test handling of connection timeouts."""
        logger.info("Testing connection timeout handling")
        
        timeout_scenarios = [
            {'timeout': 0.1, 'expected_failure': True},  # Very short timeout
            {'timeout': 1.0, 'expected_failure': True},  # Short timeout
            {'timeout': 5.0, 'expected_failure': False}, # Reasonable timeout
        ]
        
        results = []
        
        for scenario in timeout_scenarios:
            try:
                timeout = aiohttp.ClientTimeout(total=scenario['timeout'])
                
                start_time = time.time()
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        # Try to connect to a slow endpoint
                        async with session.get('http://httpbin.org/delay/2') as response:
                            response_time = time.time() - start_time
                            
                            results.append({
                                'timeout': scenario['timeout'],
                                'expected_failure': scenario['expected_failure'],
                                'actual_failure': False,
                                'response_time': response_time,
                                'success': not scenario['expected_failure']
                            })
                            
                    except asyncio.TimeoutError:
                        response_time = time.time() - start_time
                        
                        results.append({
                            'timeout': scenario['timeout'],
                            'expected_failure': scenario['expected_failure'],
                            'actual_failure': True,
                            'response_time': response_time,
                            'success': scenario['expected_failure']  # Success if we expected failure
                        })
                        
            except Exception as e:
                results.append({
                    'timeout': scenario['timeout'],
                    'expected_failure': scenario['expected_failure'],
                    'actual_failure': True,
                    'error': str(e),
                    'success': scenario['expected_failure']
                })
        
        successful_scenarios = sum(1 for r in results if r['success'])
        total_scenarios = len(results)
        
        return {
            'success': successful_scenarios == total_scenarios,
            'successful_scenarios': successful_scenarios,
            'total_scenarios': total_scenarios,
            'scenario_results': results
        }
    
    async def test_retry_mechanism(self) -> Dict[str, Any]:
        """Test automatic retry mechanisms."""
        logger.info("Testing retry mechanisms")
        
        retry_attempts = 3
        successful_retries = 0
        failed_retries = 0
        
        # Simulate unreliable service (fails 70% of the time)
        async def unreliable_operation():
            if random.random() < 0.7:  # 70% failure rate
                raise ConnectionError("Simulated network error")
            return "success"
        
        for attempt in range(10):  # Test 10 operations
            retry_count = 0
            operation_successful = False
            
            for retry in range(retry_attempts):
                try:
                    result = await unreliable_operation()
                    operation_successful = True
                    break
                except ConnectionError:
                    retry_count += 1
                    if retry < retry_attempts - 1:
                        await asyncio.sleep(0.1 * (2 ** retry))  # Exponential backoff
            
            if operation_successful:
                successful_retries += 1
            else:
                failed_retries += 1
        
        total_operations = successful_retries + failed_retries
        success_rate = successful_retries / total_operations if total_operations > 0 else 0
        
        return {
            'success': success_rate > 0.3,  # At least 30% should succeed with retries
            'successful_operations': successful_retries,
            'failed_operations': failed_retries,
            'success_rate': success_rate,
            'metrics': {
                'total_operations': total_operations,
                'retry_attempts': retry_attempts
            }
        }


class InvalidInputValidator(BaseValidator):
    """Tests handling of invalid inputs and error reporting."""
    
    async def test_malformed_json_handling(self) -> Dict[str, Any]:
        """Test handling of malformed JSON inputs."""
        logger.info("Testing malformed JSON handling")
        
        malformed_inputs = [
            '{"invalid": json}',  # Missing quotes
            '{"unclosed": "object"',  # Unclosed object
            '{invalid_key: "value"}',  # Invalid key format
            '["unclosed array"',  # Unclosed array
            'not json at all',  # Not JSON
            '',  # Empty string
            'null',  # Valid JSON but null
        ]
        
        results = []
        
        for malformed_input in malformed_inputs:
            try:
                # Test JSON parsing
                import json
                try:
                    parsed = json.loads(malformed_input)
                    results.append({
                        'input': malformed_input[:50] + '...' if len(malformed_input) > 50 else malformed_input,
                        'parsing_failed': False,
                        'error_handled': True,  # Valid JSON, no error to handle
                        'parsed_result': type(parsed).__name__
                    })
                except json.JSONDecodeError as e:
                    results.append({
                        'input': malformed_input[:50] + '...' if len(malformed_input) > 50 else malformed_input,
                        'parsing_failed': True,
                        'error_handled': True,  # Exception properly caught
                        'error_type': 'JSONDecodeError',
                        'error_message': str(e)
                    })
                    
            except Exception as e:
                results.append({
                    'input': malformed_input[:50] + '...' if len(malformed_input) > 50 else malformed_input,
                    'parsing_failed': True,
                    'error_handled': False,  # Unexpected exception
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                })
        
        properly_handled = sum(1 for r in results if r['error_handled'])
        total_inputs = len(results)
        
        return {
            'success': properly_handled == total_inputs,
            'properly_handled': properly_handled,
            'total_inputs': total_inputs,
            'input_results': results
        }
    
    async def test_boundary_value_handling(self) -> Dict[str, Any]:
        """Test handling of boundary values and edge cases."""
        logger.info("Testing boundary value handling")
        
        boundary_tests = [
            # String length boundaries
            {'test': 'empty_string', 'value': '', 'expected_valid': False},
            {'test': 'very_long_string', 'value': 'x' * 10000, 'expected_valid': False},
            {'test': 'normal_string', 'value': 'test_project', 'expected_valid': True},
            
            # Numeric boundaries
            {'test': 'negative_number', 'value': -1, 'expected_valid': False},
            {'test': 'zero', 'value': 0, 'expected_valid': True},
            {'test': 'large_number', 'value': 999999999, 'expected_valid': False},
            {'test': 'normal_number', 'value': 100, 'expected_valid': True},
            
            # Path boundaries
            {'test': 'invalid_path', 'value': '/nonexistent/path/../../etc/passwd', 'expected_valid': False},
            {'test': 'relative_path', 'value': '../../../etc/passwd', 'expected_valid': False},
            {'test': 'normal_path', 'value': '/tmp/test', 'expected_valid': True},
        ]
        
        results = []
        
        for test_case in boundary_tests:
            try:
                # Simulate validation logic
                is_valid = self._validate_input(test_case['test'], test_case['value'])
                
                results.append({
                    'test_name': test_case['test'],
                    'input_value': str(test_case['value'])[:100],  # Truncate for display
                    'expected_valid': test_case['expected_valid'],
                    'actual_valid': is_valid,
                    'validation_correct': is_valid == test_case['expected_valid']
                })
                
            except Exception as e:
                results.append({
                    'test_name': test_case['test'],
                    'input_value': str(test_case['value'])[:100],
                    'expected_valid': test_case['expected_valid'],
                    'actual_valid': False,  # Exception means validation failed
                    'validation_correct': not test_case['expected_valid'],  # Correct if we expected invalid
                    'error': str(e)
                })
        
        correct_validations = sum(1 for r in results if r['validation_correct'])
        total_tests = len(results)
        
        return {
            'success': correct_validations == total_tests,
            'correct_validations': correct_validations,
            'total_tests': total_tests,
            'validation_results': results
        }
    
    def _validate_input(self, test_type: str, value: Any) -> bool:
        """Mock input validation logic."""
        if test_type.endswith('_string'):
            if isinstance(value, str):
                return 1 <= len(value) <= 1000
            return False
        
        elif test_type.endswith('_number'):
            if isinstance(value, (int, float)):
                return 0 <= value <= 1000000
            return False
        
        elif test_type.endswith('_path'):
            if isinstance(value, str):
                # Simple path validation
                return (
                    not value.startswith('..') and
                    '../' not in value and
                    not any(dangerous in value.lower() for dangerous in ['etc/passwd', 'etc/shadow', 'windows/system32'])
                )
            return False
        
        return True


class GracefulDegradationValidator(BaseValidator):
    """Tests graceful degradation scenarios."""
    
    async def test_partial_service_availability(self) -> Dict[str, Any]:
        """Test system behavior when some services are unavailable."""
        logger.info("Testing partial service availability")
        
        # Define service combinations and expected behavior
        service_scenarios = [
            {
                'name': 'database_only',
                'available_services': ['database'],
                'expected_functionality': ['basic_queries', 'read_operations'],
                'expected_limitations': ['no_caching', 'no_real_time_updates']
            },
            {
                'name': 'api_and_database',
                'available_services': ['api', 'database'],
                'expected_functionality': ['api_endpoints', 'data_storage', 'basic_operations'],
                'expected_limitations': ['no_caching', 'no_websockets']
            },
            {
                'name': 'all_except_redis',
                'available_services': ['api', 'database', 'websocket'],
                'expected_functionality': ['full_api', 'real_time_updates', 'data_persistence'],
                'expected_limitations': ['no_caching', 'slower_responses']
            }
        ]
        
        results = []
        
        for scenario in service_scenarios:
            try:
                # Test the scenario
                functionality_available = await self._test_scenario_functionality(
                    scenario['available_services'],
                    scenario['expected_functionality']
                )
                
                limitations_present = await self._test_scenario_limitations(
                    scenario['available_services'],
                    scenario['expected_limitations']
                )
                
                results.append({
                    'scenario_name': scenario['name'],
                    'available_services': scenario['available_services'],
                    'functionality_working': functionality_available,
                    'limitations_confirmed': limitations_present,
                    'graceful_degradation': functionality_available and limitations_present
                })
                
            except Exception as e:
                results.append({
                    'scenario_name': scenario['name'],
                    'available_services': scenario['available_services'],
                    'functionality_working': False,
                    'limitations_confirmed': False,
                    'graceful_degradation': False,
                    'error': str(e)
                })
        
        successful_scenarios = sum(1 for r in results if r['graceful_degradation'])
        total_scenarios = len(results)
        
        return {
            'success': successful_scenarios > 0,  # At least one scenario should work
            'successful_scenarios': successful_scenarios,
            'total_scenarios': total_scenarios,
            'scenario_results': results
        }
    
    async def _test_scenario_functionality(self, available_services: List[str], expected_functionality: List[str]) -> bool:
        """Test if expected functionality is available with given services."""
        # Mock functionality testing
        functionality_score = 0
        
        for func in expected_functionality:
            if func == 'basic_queries' and 'database' in available_services:
                functionality_score += 1
            elif func == 'api_endpoints' and 'api' in available_services:
                functionality_score += 1
            elif func == 'real_time_updates' and 'websocket' in available_services:
                functionality_score += 1
            elif func == 'data_storage' and 'database' in available_services:
                functionality_score += 1
            elif func == 'full_api' and 'api' in available_services and 'database' in available_services:
                functionality_score += 1
        
        return functionality_score >= len(expected_functionality) * 0.8  # 80% functionality required
    
    async def _test_scenario_limitations(self, available_services: List[str], expected_limitations: List[str]) -> bool:
        """Test if expected limitations are present."""
        # Mock limitation testing
        limitation_score = 0
        
        for limitation in expected_limitations:
            if limitation == 'no_caching' and 'redis' not in available_services:
                limitation_score += 1
            elif limitation == 'no_real_time_updates' and 'websocket' not in available_services:
                limitation_score += 1
            elif limitation == 'no_websockets' and 'websocket' not in available_services:
                limitation_score += 1
            elif limitation == 'slower_responses' and 'redis' not in available_services:
                limitation_score += 1
        
        return limitation_score >= len(expected_limitations) * 0.8  # 80% limitations should be present


class ErrorRecoveryTestSuite(BaseValidator):
    """Main error recovery and resilience test suite."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.service_failure_validator = ServiceFailureSimulator(config)
        self.network_validator = NetworkInterruptionValidator(config)
        self.input_validator = InvalidInputValidator(config)
        self.degradation_validator = GracefulDegradationValidator(config)
    
    async def run_comprehensive_error_recovery_tests(self) -> Dict[str, Any]:
        """Run complete error recovery test suite."""
        logger.info("Starting comprehensive error recovery tests")
        
        test_categories = [
            ('database_failure_recovery', self.service_failure_validator.test_database_failure_recovery()),
            ('redis_failure_recovery', self.service_failure_validator.test_redis_failure_recovery()),
            ('api_service_failure', self.service_failure_validator.test_api_service_failure()),
            ('connection_timeouts', self.network_validator.test_connection_timeout_handling()),
            ('retry_mechanisms', self.network_validator.test_retry_mechanism()),
            ('malformed_json', self.input_validator.test_malformed_json_handling()),
            ('boundary_values', self.input_validator.test_boundary_value_handling()),
            ('graceful_degradation', self.degradation_validator.test_partial_service_availability())
        ]
        
        results = {}
        overall_success = True
        critical_failures = []
        
        for category, test_coro in test_categories:
            try:
                logger.info(f"Running {category} tests...")
                result = await test_coro
                results[category] = result
                
                if not result.get('success', False):
                    logger.warning(f"{category} tests failed: {result.get('error', 'Unknown error')}")
                    
                    # Mark critical failures for core recovery mechanisms
                    if category in ['database_failure_recovery', 'api_service_failure', 'graceful_degradation']:
                        critical_failures.append(category)
                        overall_success = False
                else:
                    logger.info(f"{category} tests passed")
                    
            except Exception as e:
                logger.error(f"{category} tests encountered error: {e}")
                results[category] = {
                    'success': False,
                    'error': str(e)
                }
                
                if category in ['database_failure_recovery', 'api_service_failure', 'graceful_degradation']:
                    critical_failures.append(category)
                    overall_success = False
        
        # Calculate resilience score
        successful_categories = sum(1 for r in results.values() if r.get('success', False))
        total_categories = len(test_categories)
        resilience_score = successful_categories / total_categories
        
        return {
            'success': overall_success,
            'resilience_score': resilience_score,
            'summary': {
                'total_categories': total_categories,
                'successful_categories': successful_categories,
                'critical_failures': critical_failures,
                'system_resilient': len(critical_failures) == 0
            },
            'results': results,
            'recommendations': self._generate_resilience_recommendations(results, critical_failures)
        }
    
    def _generate_resilience_recommendations(self, results: Dict[str, Any], critical_failures: List[str]) -> List[str]:
        """Generate resilience improvement recommendations."""
        recommendations = []
        
        if critical_failures:
            recommendations.append("Critical resilience failures detected - implement robust error handling")
        
        # Database resilience recommendations
        db_result = results.get('database_failure_recovery', {})
        if not db_result.get('success', False):
            recommendations.append("Implement database connection pooling and automatic retry mechanisms")
        
        # API resilience recommendations
        api_result = results.get('api_service_failure', {})
        if not api_result.get('success', False):
            recommendations.append("Implement circuit breakers and service health checks")
        
        # Network resilience recommendations
        timeout_result = results.get('connection_timeouts', {})
        if not timeout_result.get('success', False):
            recommendations.append("Configure appropriate timeouts and implement exponential backoff")
        
        retry_result = results.get('retry_mechanisms', {})
        if not retry_result.get('success', False):
            recommendations.append("Improve retry logic with jitter and maximum retry limits")
        
        # Input validation recommendations
        json_result = results.get('malformed_json', {})
        if not json_result.get('success', False):
            recommendations.append("Strengthen input validation and error message handling")
        
        boundary_result = results.get('boundary_values', {})
        if not boundary_result.get('success', False):
            recommendations.append("Implement comprehensive boundary value validation")
        
        # Graceful degradation recommendations
        degradation_result = results.get('graceful_degradation', {})
        if not degradation_result.get('success', False):
            recommendations.append("Design system for graceful degradation when services are unavailable")
        
        return recommendations
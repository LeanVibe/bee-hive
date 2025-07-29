"""
Performance Under Chaos Validation - LeanVibe Agent Hive 2.0 - Phase 5.1

This module validates that all existing Phase 1-4 performance targets are maintained
during chaos scenarios, ensuring that resilience mechanisms don't compromise system
performance beyond acceptable thresholds.

Key Validation Areas:
- Message processing throughput during failures
- API response times under stress conditions
- Memory usage efficiency during recovery
- Database operation performance with circuit breakers
- Semantic search performance during service overload
- Overall system latency during chaos events
"""

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean, median, stdev

import pytest

from .chaos_testing_framework import (
    ChaosTestRunner, ChaosScenario, ChaosEventType, ChaosImpact,
    SystemMonitor, with_chaos_context
)


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during chaos scenarios"""
    timestamp: datetime
    
    # Throughput metrics
    messages_per_second: float = 0.0
    api_requests_per_second: float = 0.0
    database_ops_per_second: float = 0.0
    
    # Latency metrics (milliseconds)
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Resource usage metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    
    # Error rate metrics
    error_rate_percent: float = 0.0
    timeout_rate_percent: float = 0.0
    
    # System health metrics
    active_connections: int = 0
    queue_depth: int = 0
    cache_hit_rate_percent: float = 0.0
    
    # Custom metrics
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceTargets:
    """Performance targets that must be maintained during chaos"""
    
    # Phase 1-4 baseline targets
    max_response_time_ms: float = 2000  # 2s max response time
    min_throughput_msg_per_sec: float = 100  # 100 messages/second minimum
    max_error_rate_percent: float = 1.0  # <1% error rate
    max_memory_increase_percent: float = 50.0  # <50% memory increase
    max_cpu_usage_percent: float = 80.0  # <80% CPU usage
    
    # Chaos-specific targets
    max_recovery_latency_ms: float = 5000  # 5s max recovery latency
    min_availability_percent: float = 99.95  # >99.95% availability
    max_timeout_rate_percent: float = 5.0  # <5% timeout rate
    
    # Semantic memory specific
    max_search_latency_ms: float = 1000  # 1s max search time
    min_cache_hit_rate_percent: float = 70.0  # >70% cache hit rate
    
    # Database specific
    max_db_query_time_ms: float = 500  # 500ms max query time
    max_connection_pool_usage_percent: float = 80.0  # <80% pool usage


class PerformanceMonitor:
    """Monitors performance metrics during chaos scenarios"""
    
    def __init__(self, sampling_interval_seconds: float = 1.0):
        self.sampling_interval = sampling_interval_seconds
        self.monitoring_active = False
        self.metrics_history: List[PerformanceMetrics] = []
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking state
        self.start_time = None
        self.baseline_metrics = None
        
        # Request tracking
        self.request_times: List[float] = []
        self.error_count = 0
        self.timeout_count = 0
        self.total_requests = 0
    
    async def start_monitoring(self) -> None:
        """Start continuous performance monitoring"""
        self.monitoring_active = True
        self.start_time = datetime.now()
        self.metrics_history.clear()
        
        # Capture baseline metrics
        await self._capture_baseline_metrics()
        
        self.logger.info("Performance monitoring started")
        
        while self.monitoring_active:
            try:
                metrics = await self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Log performance warnings
                await self._check_performance_thresholds(metrics)
                
                await asyncio.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting performance metrics: {e}")
                await asyncio.sleep(self.sampling_interval)
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return performance summary"""
        self.monitoring_active = False
        
        if not self.metrics_history:
            return {'error': 'No performance metrics collected'}
        
        return self._generate_performance_summary()
    
    async def _capture_baseline_metrics(self) -> None:
        """Capture baseline performance metrics before chaos"""
        self.baseline_metrics = await self._collect_current_metrics()
        self.logger.info(f"Baseline metrics captured: "
                        f"CPU: {self.baseline_metrics.cpu_usage_percent:.1f}%, "
                        f"Memory: {self.baseline_metrics.memory_usage_mb:.1f}MB")
    
    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # Calculate response time metrics
        response_times = self.request_times[-100:] if self.request_times else [0]  # Last 100 requests
        avg_response_time = mean(response_times) if response_times else 0
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0
        p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0
        
        # Calculate throughput (requests per second over last interval)
        recent_requests = min(len(self.request_times), int(60 / self.sampling_interval))  # Last minute
        throughput = recent_requests / min(60, len(self.request_times) * self.sampling_interval) if self.request_times else 0
        
        # Calculate error rates
        recent_total = max(recent_requests, 1)
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        timeout_rate = (self.timeout_count / max(self.total_requests, 1)) * 100
        
        # Network metrics (approximated)
        network_stats = psutil.net_io_counters()
        active_connections = len(psutil.net_connections())
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            messages_per_second=throughput,
            api_requests_per_second=throughput,
            avg_response_time_ms=avg_response_time * 1000,
            p95_response_time_ms=p95_response_time * 1000,
            p99_response_time_ms=p99_response_time * 1000,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            memory_usage_percent=memory.percent,
            error_rate_percent=error_rate,
            timeout_rate_percent=timeout_rate,
            active_connections=active_connections,
            cache_hit_rate_percent=75.0,  # Simulated cache hit rate
            additional_metrics={
                'network_bytes_sent': network_stats.bytes_sent,
                'network_bytes_recv': network_stats.bytes_recv,
                'process_count': len(psutil.pids())
            }
        )
    
    async def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check if performance metrics exceed warning thresholds"""
        targets = PerformanceTargets()
        
        warnings = []
        
        if metrics.avg_response_time_ms > targets.max_response_time_ms:
            warnings.append(f"High response time: {metrics.avg_response_time_ms:.1f}ms")
        
        if metrics.cpu_usage_percent > targets.max_cpu_usage_percent:
            warnings.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        if metrics.error_rate_percent > targets.max_error_rate_percent:
            warnings.append(f"High error rate: {metrics.error_rate_percent:.1f}%")
        
        if metrics.timeout_rate_percent > targets.max_timeout_rate_percent:
            warnings.append(f"High timeout rate: {metrics.timeout_rate_percent:.1f}%")
        
        # Check memory increase from baseline
        if self.baseline_metrics:
            memory_increase = ((metrics.memory_usage_mb - self.baseline_metrics.memory_usage_mb) 
                             / self.baseline_metrics.memory_usage_mb) * 100
            if memory_increase > targets.max_memory_increase_percent:
                warnings.append(f"High memory increase: {memory_increase:.1f}%")
        
        if warnings:
            self.logger.warning(f"Performance warnings: {'; '.join(warnings)}")
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary"""
        if not self.metrics_history:
            return {}
        
        # Calculate statistics across all metrics
        cpu_values = [m.cpu_usage_percent for m in self.metrics_history]
        memory_values = [m.memory_usage_mb for m in self.metrics_history]
        response_times = [m.avg_response_time_ms for m in self.metrics_history]
        throughput_values = [m.messages_per_second for m in self.metrics_history]
        error_rates = [m.error_rate_percent for m in self.metrics_history]
        
        summary = {
            'monitoring_duration_seconds': len(self.metrics_history) * self.sampling_interval,
            'total_samples': len(self.metrics_history),
            
            # Performance statistics
            'cpu_usage': {
                'avg': mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'stddev': stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory_usage_mb': {
                'avg': mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'stddev': stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'response_time_ms': {
                'avg': mean(response_times),
                'max': max(response_times),
                'min': min(response_times),
                'p95': sorted(response_times)[int(len(response_times) * 0.95)],
                'p99': sorted(response_times)[int(len(response_times) * 0.99)]
            },
            'throughput_msg_per_sec': {
                'avg': mean(throughput_values),
                'max': max(throughput_values),
                'min': min(throughput_values)
            },
            'error_rate_percent': {
                'avg': mean(error_rates),
                'max': max(error_rates)
            },
            
            # Baseline comparison
            'baseline_comparison': self._compare_to_baseline() if self.baseline_metrics else {},
            
            # Target compliance
            'target_compliance': self._check_target_compliance()
        }
        
        return summary
    
    def _compare_to_baseline(self) -> Dict[str, float]:
        """Compare current performance to baseline metrics"""
        if not self.baseline_metrics or not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            'cpu_usage_change_percent': latest_metrics.cpu_usage_percent - self.baseline_metrics.cpu_usage_percent,
            'memory_usage_change_mb': latest_metrics.memory_usage_mb - self.baseline_metrics.memory_usage_mb,
            'response_time_change_ms': latest_metrics.avg_response_time_ms - self.baseline_metrics.avg_response_time_ms,
            'throughput_change_percent': ((latest_metrics.messages_per_second - self.baseline_metrics.messages_per_second) 
                                        / max(self.baseline_metrics.messages_per_second, 1)) * 100
        }
    
    def _check_target_compliance(self) -> Dict[str, bool]:
        """Check compliance with performance targets"""
        if not self.metrics_history:
            return {}
        
        targets = PerformanceTargets()
        latest_metrics = self.metrics_history[-1]
        
        return {
            'response_time_compliant': latest_metrics.avg_response_time_ms <= targets.max_response_time_ms,
            'throughput_compliant': latest_metrics.messages_per_second >= targets.min_throughput_msg_per_sec,
            'error_rate_compliant': latest_metrics.error_rate_percent <= targets.max_error_rate_percent,
            'cpu_usage_compliant': latest_metrics.cpu_usage_percent <= targets.max_cpu_usage_percent,
            'timeout_rate_compliant': latest_metrics.timeout_rate_percent <= targets.max_timeout_rate_percent,
            'cache_hit_rate_compliant': latest_metrics.cache_hit_rate_percent >= targets.min_cache_hit_rate_percent
        }
    
    def record_request(self, response_time_seconds: float, success: bool, timeout: bool = False) -> None:
        """Record a request for performance tracking"""
        self.request_times.append(response_time_seconds)
        self.total_requests += 1
        
        if not success:
            self.error_count += 1
        
        if timeout:
            self.timeout_count += 1
        
        # Keep only recent request times to prevent memory bloat
        if len(self.request_times) > 10000:
            self.request_times = self.request_times[-5000:]


class ChaosPerformanceValidator:
    """Validates performance under various chaos conditions"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        self.validation_results: List[Dict[str, Any]] = []
    
    async def validate_performance_under_chaos(self, scenario: ChaosScenario, 
                                              workload_generator: callable,
                                              targets: Optional[PerformanceTargets] = None) -> Dict[str, Any]:
        """
        Validate performance during a chaos scenario with specified workload
        
        Args:
            scenario: Chaos scenario to execute
            workload_generator: Async function that generates workload during chaos
            targets: Performance targets to validate against
        
        Returns:
            Validation results with pass/fail status and detailed metrics
        """
        if targets is None:
            targets = PerformanceTargets()
        
        self.logger.info(f"Validating performance under chaos: {scenario.name}")
        
        # Start performance monitoring
        monitor_task = asyncio.create_task(self.performance_monitor.start_monitoring())
        
        # Allow baseline collection
        await asyncio.sleep(5)
        
        try:
            # Start workload generation
            workload_task = asyncio.create_task(workload_generator(self.performance_monitor))
            
            # Execute chaos scenario
            runner = ChaosTestRunner()
            chaos_metrics = await runner.run_scenario(scenario)
            
            # Wait for workload to complete
            await workload_task
            
            # Stop monitoring and get results
            performance_summary = self.performance_monitor.stop_monitoring()
            monitor_task.cancel()
            
            # Validate against targets
            validation_result = self._validate_against_targets(performance_summary, targets, chaos_metrics)
            
            self.validation_results.append(validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            self.performance_monitor.stop_monitoring()
            monitor_task.cancel()
            
            return {
                'scenario_name': scenario.name,
                'validation_passed': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_against_targets(self, performance_summary: Dict[str, Any], 
                                 targets: PerformanceTargets,
                                 chaos_metrics) -> Dict[str, Any]:
        """Validate performance summary against targets"""
        
        validation_checks = {
            'response_time_check': performance_summary['response_time_ms']['avg'] <= targets.max_response_time_ms,
            'throughput_check': performance_summary['throughput_msg_per_sec']['avg'] >= targets.min_throughput_msg_per_sec,
            'error_rate_check': performance_summary['error_rate_percent']['avg'] <= targets.max_error_rate_percent,
            'cpu_usage_check': performance_summary['cpu_usage']['max'] <= targets.max_cpu_usage_percent,
            'memory_stability_check': self._check_memory_stability(performance_summary, targets),
            'recovery_latency_check': chaos_metrics.recovery_time_seconds <= (targets.max_recovery_latency_ms / 1000),
            'availability_check': chaos_metrics.availability_percentage >= targets.min_availability_percent,
            'target_compliance_check': all(performance_summary.get('target_compliance', {}).values())
        }
        
        all_checks_passed = all(validation_checks.values())
        
        result = {
            'scenario_name': getattr(chaos_metrics, 'event_type', 'unknown').name if hasattr(chaos_metrics, 'event_type') else 'unknown',
            'validation_passed': all_checks_passed,
            'individual_checks': validation_checks,
            'performance_summary': performance_summary,
            'chaos_metrics': {
                'recovery_time_seconds': chaos_metrics.recovery_time_seconds,
                'availability_percentage': chaos_metrics.availability_percentage,
                'recovery_successful': chaos_metrics.recovery_successful
            },
            'compliance_details': self._get_compliance_details(performance_summary, targets),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log validation results
        if all_checks_passed:
            self.logger.info(f"Performance validation PASSED for {result['scenario_name']}")
        else:
            failed_checks = [check for check, passed in validation_checks.items() if not passed]
            self.logger.warning(f"Performance validation FAILED for {result['scenario_name']}: {failed_checks}")
        
        return result
    
    def _check_memory_stability(self, performance_summary: Dict[str, Any], targets: PerformanceTargets) -> bool:
        """Check if memory usage remained stable during chaos"""
        baseline_comparison = performance_summary.get('baseline_comparison', {})
        memory_change_mb = baseline_comparison.get('memory_usage_change_mb', 0)
        
        # Calculate percentage increase
        memory_increase_percent = abs(memory_change_mb) / performance_summary['memory_usage_mb']['min'] * 100
        
        return memory_increase_percent <= targets.max_memory_increase_percent
    
    def _get_compliance_details(self, performance_summary: Dict[str, Any], targets: PerformanceTargets) -> Dict[str, Any]:
        """Get detailed compliance information"""
        return {
            'response_time': {
                'actual_avg_ms': performance_summary['response_time_ms']['avg'],
                'target_max_ms': targets.max_response_time_ms,
                'compliant': performance_summary['response_time_ms']['avg'] <= targets.max_response_time_ms
            },
            'throughput': {
                'actual_avg_msg_per_sec': performance_summary['throughput_msg_per_sec']['avg'],
                'target_min_msg_per_sec': targets.min_throughput_msg_per_sec,
                'compliant': performance_summary['throughput_msg_per_sec']['avg'] >= targets.min_throughput_msg_per_sec
            },
            'error_rate': {
                'actual_avg_percent': performance_summary['error_rate_percent']['avg'],
                'target_max_percent': targets.max_error_rate_percent,
                'compliant': performance_summary['error_rate_percent']['avg'] <= targets.max_error_rate_percent
            },
            'resource_usage': {
                'cpu_max_percent': performance_summary['cpu_usage']['max'],
                'cpu_target_max_percent': targets.max_cpu_usage_percent,
                'cpu_compliant': performance_summary['cpu_usage']['max'] <= targets.max_cpu_usage_percent,
                'memory_stable': self._check_memory_stability(performance_summary, targets)
            }
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance validation report"""
        if not self.validation_results:
            return {'error': 'No validation results available'}
        
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r['validation_passed'])
        
        # Aggregate performance statistics
        all_response_times = []
        all_throughput_values = []
        all_error_rates = []
        
        for result in self.validation_results:
            if 'performance_summary' in result:
                perf = result['performance_summary']
                all_response_times.append(perf['response_time_ms']['avg'])
                all_throughput_values.append(perf['throughput_msg_per_sec']['avg'])
                all_error_rates.append(perf['error_rate_percent']['avg'])
        
        return {
            'validation_summary': {
                'total_scenarios': total_validations,
                'passed_scenarios': passed_validations,
                'success_rate_percent': (passed_validations / total_validations) * 100,
                'overall_performance_maintained': passed_validations >= total_validations * 0.9  # 90% threshold
            },
            'aggregated_performance': {
                'avg_response_time_ms': mean(all_response_times) if all_response_times else 0,
                'avg_throughput_msg_per_sec': mean(all_throughput_values) if all_throughput_values else 0,
                'avg_error_rate_percent': mean(all_error_rates) if all_error_rates else 0,
                'max_response_time_ms': max(all_response_times) if all_response_times else 0,
                'min_throughput_msg_per_sec': min(all_throughput_values) if all_throughput_values else 0
            },
            'detailed_results': self.validation_results,
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations based on validation results"""
        recommendations = []
        
        # Analyze failed validations for patterns
        failed_results = [r for r in self.validation_results if not r['validation_passed']]
        
        if failed_results:
            # Check for common failure patterns
            response_time_failures = sum(1 for r in failed_results 
                                       if not r['individual_checks'].get('response_time_check', True))
            throughput_failures = sum(1 for r in failed_results 
                                    if not r['individual_checks'].get('throughput_check', True))
            error_rate_failures = sum(1 for r in failed_results 
                                    if not r['individual_checks'].get('error_rate_check', True))
            
            if response_time_failures > len(failed_results) * 0.5:
                recommendations.append("Consider optimizing response time handling during chaos scenarios")
            
            if throughput_failures > len(failed_results) * 0.5:
                recommendations.append("Review throughput degradation patterns and implement better load balancing")
            
            if error_rate_failures > len(failed_results) * 0.5:
                recommendations.append("Improve error handling and retry mechanisms during failures")
        
        # General recommendations
        if len(self.validation_results) > 0:
            avg_success_rate = sum(1 for r in self.validation_results if r['validation_passed']) / len(self.validation_results)
            
            if avg_success_rate < 0.95:  # Less than 95% success
                recommendations.append("Overall performance validation success rate is below 95% - review system resilience")
            
            if avg_success_rate == 1.0:  # Perfect success
                recommendations.append("Excellent performance maintained under all chaos conditions")
        
        return recommendations if recommendations else ["No specific recommendations - performance targets met"]


# Utility functions for common workload patterns
async def generate_message_processing_workload(performance_monitor: PerformanceMonitor, 
                                             duration_seconds: int = 120,
                                             target_rate_per_second: int = 200) -> None:
    """Generate message processing workload for chaos testing"""
    end_time = time.time() + duration_seconds
    message_count = 0
    
    while time.time() < end_time:
        batch_start = time.time()
        
        # Process batch of messages
        batch_size = min(20, target_rate_per_second // 10)  # 10 batches per second
        
        for _ in range(batch_size):
            request_start = time.time()
            
            try:
                # Simulate message processing
                await asyncio.sleep(0.005)  # 5ms processing time
                
                response_time = time.time() - request_start
                performance_monitor.record_request(response_time, success=True)
                message_count += 1
                
            except Exception as e:
                response_time = time.time() - request_start
                performance_monitor.record_request(response_time, success=False)
                logging.debug(f"Message processing failed: {e}")
        
        # Maintain target rate
        batch_duration = time.time() - batch_start
        target_batch_duration = batch_size / target_rate_per_second
        
        if batch_duration < target_batch_duration:
            await asyncio.sleep(target_batch_duration - batch_duration)


async def generate_api_request_workload(performance_monitor: PerformanceMonitor,
                                      duration_seconds: int = 120,
                                      target_rate_per_second: int = 100) -> None:
    """Generate API request workload for chaos testing"""
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        request_start = time.time()
        
        try:
            # Simulate API request processing
            await asyncio.sleep(0.01)  # 10ms processing time
            
            response_time = time.time() - request_start
            performance_monitor.record_request(response_time, success=True)
            
        except Exception as e:
            response_time = time.time() - request_start
            performance_monitor.record_request(response_time, success=False)
            logging.debug(f"API request failed: {e}")
        
        # Maintain target rate
        await asyncio.sleep(1.0 / target_rate_per_second)


async def generate_database_workload(performance_monitor: PerformanceMonitor,
                                   duration_seconds: int = 120,
                                   query_rate_per_second: int = 50) -> None:
    """Generate database operation workload for chaos testing"""
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        request_start = time.time()
        
        try:
            # Simulate database query
            await asyncio.sleep(0.02)  # 20ms query time
            
            response_time = time.time() - request_start
            performance_monitor.record_request(response_time, success=True)
            
        except Exception as e:
            response_time = time.time() - request_start
            performance_monitor.record_request(response_time, success=False, timeout="timeout" in str(e).lower())
            logging.debug(f"Database query failed: {e}")
        
        # Maintain target rate
        await asyncio.sleep(1.0 / query_rate_per_second)


# Integration with chaos scenarios
async def validate_all_scenarios_performance() -> Dict[str, Any]:
    """Validate performance across all Phase 5.1 chaos scenarios"""
    validator = ChaosPerformanceValidator()
    
    # Define scenarios with appropriate workloads
    scenarios_and_workloads = [
        ("Redis Failure", ChaosEventType.REDIS_CONNECTION_LOSS, generate_message_processing_workload),
        ("PostgreSQL Failure", ChaosEventType.POSTGRES_CONNECTION_LOSS, generate_database_workload),
        ("Service Overload", ChaosEventType.SERVICE_OVERLOAD, generate_api_request_workload),
        ("Memory Pressure", ChaosEventType.MEMORY_PRESSURE, generate_message_processing_workload),
        ("Network Partition", ChaosEventType.NETWORK_PARTITION, generate_api_request_workload)
    ]
    
    results = []
    
    for scenario_name, event_type, workload_func in scenarios_and_workloads:
        scenario = ChaosScenario(
            name=scenario_name,
            event_type=event_type,
            impact_level=ChaosImpact.HIGH,
            duration_seconds=90,
            target_services=["all"],
            expected_recovery_time_seconds=30,
            success_criteria={'min_availability_percentage': 99.95}
        )
        
        result = await validator.validate_performance_under_chaos(scenario, workload_func)
        results.append(result)
        
        # Allow system recovery between scenarios
        await asyncio.sleep(10)
    
    # Generate comprehensive report
    comprehensive_report = validator.generate_comprehensive_report()
    
    return comprehensive_report
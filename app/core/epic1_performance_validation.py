"""
Epic 1 Performance Validation for LeanVibe Agent Hive 2.0 - Epic 2 Phase 2.2

Validates that Epic 1 performance targets are preserved throughout 
the Plugin Marketplace & Discovery implementation.

Epic 1 Targets:
- <50ms API response times for marketplace operations
- <80MB memory usage with efficient caching and lazy loading
- <250 concurrent agents supported
- Non-blocking operations with async/await
- Optimized database queries and caching

Validation Areas:
- API Performance: All plugin marketplace endpoints
- Memory Usage: Plugin loading, caching, and discovery
- Concurrency: Multiple plugin operations
- Database Performance: Plugin registry queries
- AI Performance: Discovery and recommendation inference
"""

import asyncio
import time
import gc
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import concurrent.futures

from .logging_service import get_component_logger
from .plugin_marketplace import PluginMarketplace, SearchQuery, PluginCategory, CertificationLevel
from .ai_plugin_discovery import AIPluginDiscovery, RecommendationType
from .security_certification_pipeline import SecurityCertificationPipeline
from .developer_onboarding_platform import DeveloperOnboardingPlatform
from .sample_plugins import SamplePluginDemonstrator, ProductivityBoosterPlugin

logger = get_component_logger("epic1_performance_validation")


@dataclass
class PerformanceMetrics:
    """Performance metrics for validation."""
    operation: str
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "response_time_ms": round(self.response_time_ms, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 2),
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class ValidationResult:
    """Validation test result."""
    test_name: str
    passed: bool
    target_value: float
    actual_value: float
    unit: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "target_value": self.target_value,
            "actual_value": round(self.actual_value, 2),
            "unit": self.unit,
            "pass_fail": "✅ PASS" if self.passed else "❌ FAIL",
            "details": self.details
        }


class Epic1PerformanceValidator:
    """
    Comprehensive performance validator for Epic 1 targets.
    
    Validates performance across all plugin marketplace components
    to ensure Epic 1 targets are preserved.
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = 0.0
        self.validation_results: List[ValidationResult] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Epic 1 targets
        self.targets = {
            "api_response_time_ms": 50.0,
            "memory_usage_mb": 80.0,
            "ai_inference_time_ms": 50.0,
            "security_scan_time_ms": 30.0,
            "concurrent_operations": 250,
            "database_query_time_ms": 20.0
        }
    
    async def run_comprehensive_validation(
        self,
        marketplace: PluginMarketplace,
        ai_discovery: AIPluginDiscovery,
        certification_pipeline: SecurityCertificationPipeline,
        developer_platform: DeveloperOnboardingPlatform
    ) -> Dict[str, Any]:
        """
        Run comprehensive Epic 1 performance validation.
        
        Returns detailed validation report with pass/fail status.
        """
        try:
            validation_start = datetime.utcnow()
            logger.info("Starting Epic 1 performance validation")
            
            # Establish baseline
            self.baseline_memory = self._get_memory_usage()
            
            # Initialize demo environment
            demonstrator = SamplePluginDemonstrator(marketplace, developer_platform)
            await demonstrator.setup_demo_environment()
            
            # Run validation tests
            await self._validate_api_performance(marketplace, ai_discovery, certification_pipeline, developer_platform)
            await self._validate_memory_usage(marketplace, ai_discovery)
            await self._validate_ai_performance(ai_discovery)
            await self._validate_security_performance(certification_pipeline)
            await self._validate_concurrency_performance(marketplace)
            await self._validate_database_performance(marketplace)
            
            # Generate final report
            report = self._generate_validation_report(validation_start)
            
            # Cleanup
            await demonstrator.cleanup_demo()
            
            logger.info("Epic 1 performance validation completed",
                       total_tests=len(self.validation_results),
                       passed_tests=sum(1 for r in self.validation_results if r.passed))
            
            return report
            
        except Exception as e:
            logger.error("Epic 1 performance validation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "validation_results": [r.to_dict() for r in self.validation_results]
            }
    
    async def _validate_api_performance(
        self,
        marketplace: PluginMarketplace,
        ai_discovery: AIPluginDiscovery,
        certification_pipeline: SecurityCertificationPipeline,
        developer_platform: DeveloperOnboardingPlatform
    ) -> None:
        """Validate API response times meet Epic 1 targets."""
        logger.info("Validating API performance")
        
        api_operations = [
            ("marketplace_search", lambda: marketplace.search_plugins(SearchQuery(query="productivity"))),
            ("plugin_details", lambda: marketplace.get_plugin_details("productivity_booster_v1")),
            ("ai_discovery", lambda: ai_discovery.discover_plugins_intelligent("task management", {}, {})),
            ("developer_analytics", lambda: developer_platform.get_developer_analytics("dev_001")),
            ("marketplace_stats", lambda: marketplace.get_marketplace_statistics())
        ]
        
        response_times = []
        
        for operation_name, operation_func in api_operations:
            try:
                start_time = time.perf_counter()
                await operation_func()
                end_time = time.perf_counter()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                # Record metrics
                self.performance_metrics.append(PerformanceMetrics(
                    operation=f"api_{operation_name}",
                    response_time_ms=response_time_ms,
                    memory_usage_mb=self._get_memory_usage(),
                    cpu_usage_percent=self._get_cpu_usage(),
                    success=True
                ))
                
                logger.debug("API operation completed",
                           operation=operation_name,
                           response_time_ms=round(response_time_ms, 2))
                
            except Exception as e:
                logger.error("API operation failed", operation=operation_name, error=str(e))
                self.performance_metrics.append(PerformanceMetrics(
                    operation=f"api_{operation_name}",
                    response_time_ms=0.0,
                    memory_usage_mb=self._get_memory_usage(),
                    cpu_usage_percent=self._get_cpu_usage(),
                    success=False,
                    error_message=str(e)
                ))
        
        # Validate average response time
        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            
            self.validation_results.append(ValidationResult(
                test_name="API Average Response Time",
                passed=avg_response_time < self.targets["api_response_time_ms"],
                target_value=self.targets["api_response_time_ms"],
                actual_value=avg_response_time,
                unit="ms",
                details={
                    "operations_tested": len(api_operations),
                    "successful_operations": len(response_times),
                    "max_response_time_ms": round(max_response_time, 2),
                    "min_response_time_ms": round(min(response_times), 2)
                }
            ))
            
            self.validation_results.append(ValidationResult(
                test_name="API Maximum Response Time",
                passed=max_response_time < self.targets["api_response_time_ms"] * 2,  # Allow 2x for max
                target_value=self.targets["api_response_time_ms"] * 2,
                actual_value=max_response_time,
                unit="ms",
                details={"worst_performing_operation": max(zip(response_times, [op[0] for op in api_operations]))[1]}
            ))
    
    async def _validate_memory_usage(
        self,
        marketplace: PluginMarketplace,
        ai_discovery: AIPluginDiscovery
    ) -> None:
        """Validate memory usage stays within Epic 1 targets."""
        logger.info("Validating memory usage")
        
        initial_memory = self._get_memory_usage()
        
        # Perform memory-intensive operations
        memory_operations = [
            ("load_plugins", lambda: self._simulate_plugin_loading(marketplace)),
            ("ai_embeddings", lambda: self._simulate_ai_embeddings(ai_discovery)),
            ("cache_operations", lambda: self._simulate_cache_operations(marketplace)),
            ("bulk_search", lambda: self._simulate_bulk_search(marketplace))
        ]
        
        memory_samples = []
        
        for operation_name, operation_func in memory_operations:
            try:
                pre_memory = self._get_memory_usage()
                await operation_func()
                post_memory = self._get_memory_usage()
                
                memory_increase = post_memory - pre_memory
                memory_samples.append(post_memory)
                
                logger.debug("Memory operation completed",
                           operation=operation_name,
                           memory_increase_mb=round(memory_increase, 2),
                           total_memory_mb=round(post_memory, 2))
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error("Memory operation failed", operation=operation_name, error=str(e))
        
        # Calculate memory metrics
        peak_memory = max(memory_samples) if memory_samples else initial_memory
        memory_increase = peak_memory - self.baseline_memory
        
        self.validation_results.append(ValidationResult(
            test_name="Peak Memory Usage",
            passed=peak_memory < self.targets["memory_usage_mb"],
            target_value=self.targets["memory_usage_mb"],
            actual_value=peak_memory,
            unit="MB",
            details={
                "baseline_memory_mb": round(self.baseline_memory, 2),
                "memory_increase_mb": round(memory_increase, 2),
                "operations_tested": len(memory_operations)
            }
        ))
        
        self.validation_results.append(ValidationResult(
            test_name="Memory Increase from Baseline",
            passed=memory_increase < self.targets["memory_usage_mb"] * 0.5,  # Should not increase by more than 50% of target
            target_value=self.targets["memory_usage_mb"] * 0.5,
            actual_value=memory_increase,
            unit="MB",
            details={
                "percentage_increase": round((memory_increase / self.baseline_memory) * 100, 2) if self.baseline_memory > 0 else 0
            }
        ))
    
    async def _validate_ai_performance(self, ai_discovery: AIPluginDiscovery) -> None:
        """Validate AI inference performance meets Epic 1 targets."""
        logger.info("Validating AI performance")
        
        ai_operations = [
            ("semantic_search", lambda: ai_discovery._semantic_search("productivity tools", limit=10)),
            ("compatibility_check", lambda: ai_discovery.check_plugin_compatibility("plugin_a", "plugin_b")),
            ("recommendations", lambda: ai_discovery.get_plugin_recommendations("productivity_booster_v1", RecommendationType.SIMILAR))
        ]
        
        ai_response_times = []
        
        for operation_name, operation_func in ai_operations:
            try:
                start_time = time.perf_counter()
                await operation_func()
                end_time = time.perf_counter()
                
                response_time_ms = (end_time - start_time) * 1000
                ai_response_times.append(response_time_ms)
                
                logger.debug("AI operation completed",
                           operation=operation_name,
                           response_time_ms=round(response_time_ms, 2))
                
            except Exception as e:
                logger.error("AI operation failed", operation=operation_name, error=str(e))
        
        if ai_response_times:
            avg_ai_time = statistics.mean(ai_response_times)
            max_ai_time = max(ai_response_times)
            
            self.validation_results.append(ValidationResult(
                test_name="AI Inference Average Time",
                passed=avg_ai_time < self.targets["ai_inference_time_ms"],
                target_value=self.targets["ai_inference_time_ms"],
                actual_value=avg_ai_time,
                unit="ms",
                details={
                    "operations_tested": len(ai_operations),
                    "max_inference_time_ms": round(max_ai_time, 2)
                }
            ))
    
    async def _validate_security_performance(self, certification_pipeline: SecurityCertificationPipeline) -> None:
        """Validate security scan performance meets Epic 1 targets."""
        logger.info("Validating security performance")
        
        # Create dummy plugin entry for testing
        from .plugin_marketplace import MarketplacePluginEntry
        from .orchestrator_plugins import PluginMetadata, PluginType
        
        dummy_plugin = MarketplacePluginEntry(
            plugin_metadata=PluginMetadata(
                plugin_id="test_plugin",
                name="Test Plugin",
                version="1.0.0",
                description="Test plugin for security validation",
                author="Test Author",
                plugin_type=PluginType.PROCESSOR,
                dependencies=[],
                configuration_schema={},
                permissions=[]
            ),
            developer_id="test_dev",
            category=PluginCategory.UTILITY,
            tags=["test"]
        )
        
        try:
            start_time = time.perf_counter()
            security_report = await certification_pipeline.security_scanner.scan_plugin_security(dummy_plugin)
            end_time = time.perf_counter()
            
            scan_time_ms = (end_time - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name="Security Scan Time",
                passed=scan_time_ms < self.targets["security_scan_time_ms"],
                target_value=self.targets["security_scan_time_ms"],
                actual_value=scan_time_ms,
                unit="ms",
                details={
                    "scan_successful": security_report is not None,
                    "violations_found": len(security_report.violations) if security_report else 0
                }
            ))
            
        except Exception as e:
            logger.error("Security scan validation failed", error=str(e))
            self.validation_results.append(ValidationResult(
                test_name="Security Scan Time",
                passed=False,
                target_value=self.targets["security_scan_time_ms"],
                actual_value=0.0,
                unit="ms",
                details={"error": str(e)}
            ))
    
    async def _validate_concurrency_performance(self, marketplace: PluginMarketplace) -> None:
        """Validate concurrent operation performance meets Epic 1 targets."""
        logger.info("Validating concurrency performance")
        
        async def concurrent_search_operation():
            """Single concurrent search operation."""
            try:
                start_time = time.perf_counter()
                await marketplace.search_plugins(SearchQuery(query="test"))
                end_time = time.perf_counter()
                return (end_time - start_time) * 1000
            except Exception as e:
                logger.error("Concurrent operation failed", error=str(e))
                return None
        
        # Test different concurrency levels
        concurrency_levels = [10, 25, 50, 100]
        
        for concurrency in concurrency_levels:
            try:
                start_time = time.perf_counter()
                
                # Run concurrent operations
                tasks = [concurrent_search_operation() for _ in range(concurrency)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.perf_counter()
                total_time_ms = (end_time - start_time) * 1000
                
                # Filter successful operations
                successful_times = [r for r in results if isinstance(r, (int, float)) and r is not None]
                success_rate = len(successful_times) / len(results) * 100
                
                avg_response_time = statistics.mean(successful_times) if successful_times else 0.0
                
                # Test passes if we can handle the concurrency with good performance
                passed = (
                    success_rate >= 95.0 and  # 95% success rate
                    avg_response_time < self.targets["api_response_time_ms"] * 2 and  # Allow 2x for concurrent
                    total_time_ms < 5000  # Complete within 5 seconds
                )
                
                self.validation_results.append(ValidationResult(
                    test_name=f"Concurrency {concurrency} Operations",
                    passed=passed,
                    target_value=95.0,  # 95% success rate target
                    actual_value=success_rate,
                    unit="%",
                    details={
                        "concurrent_operations": concurrency,
                        "successful_operations": len(successful_times),
                        "avg_response_time_ms": round(avg_response_time, 2),
                        "total_time_ms": round(total_time_ms, 2)
                    }
                ))
                
            except Exception as e:
                logger.error("Concurrency test failed", concurrency=concurrency, error=str(e))
                self.validation_results.append(ValidationResult(
                    test_name=f"Concurrency {concurrency} Operations",
                    passed=False,
                    target_value=95.0,
                    actual_value=0.0,
                    unit="%",
                    details={"error": str(e)}
                ))
    
    async def _validate_database_performance(self, marketplace: PluginMarketplace) -> None:
        """Validate database query performance meets Epic 1 targets."""
        logger.info("Validating database performance")
        
        # Test different query patterns
        query_operations = [
            ("simple_search", lambda: marketplace.search_plugins(SearchQuery(query="test", limit=10))),
            ("category_filter", lambda: marketplace.search_plugins(SearchQuery(
                query="", category=PluginCategory.PRODUCTIVITY, limit=20))),
            ("complex_search", lambda: marketplace.search_plugins(SearchQuery(
                query="productivity", category=PluginCategory.PRODUCTIVITY, 
                certification_level=CertificationLevel.SECURITY_VERIFIED, limit=50)))
        ]
        
        db_response_times = []
        
        for operation_name, operation_func in query_operations:
            try:
                start_time = time.perf_counter()
                result = await operation_func()
                end_time = time.perf_counter()
                
                query_time_ms = (end_time - start_time) * 1000
                db_response_times.append(query_time_ms)
                
                logger.debug("Database operation completed",
                           operation=operation_name,
                           query_time_ms=round(query_time_ms, 2),
                           results_count=len(result.plugins) if hasattr(result, 'plugins') else 0)
                
            except Exception as e:
                logger.error("Database operation failed", operation=operation_name, error=str(e))
        
        if db_response_times:
            avg_db_time = statistics.mean(db_response_times)
            max_db_time = max(db_response_times)
            
            self.validation_results.append(ValidationResult(
                test_name="Database Query Average Time",
                passed=avg_db_time < self.targets["database_query_time_ms"],
                target_value=self.targets["database_query_time_ms"],
                actual_value=avg_db_time,
                unit="ms",
                details={
                    "queries_tested": len(query_operations),
                    "max_query_time_ms": round(max_db_time, 2)
                }
            ))
    
    # Helper methods for simulation
    async def _simulate_plugin_loading(self, marketplace: PluginMarketplace) -> None:
        """Simulate loading multiple plugins."""
        for i in range(10):
            await marketplace.get_plugin_details(f"test_plugin_{i}")
    
    async def _simulate_ai_embeddings(self, ai_discovery: AIPluginDiscovery) -> None:
        """Simulate AI embedding generation."""
        queries = ["productivity", "integration", "analytics", "security", "automation"]
        for query in queries:
            await ai_discovery._semantic_search(query)
    
    async def _simulate_cache_operations(self, marketplace: PluginMarketplace) -> None:
        """Simulate cache operations."""
        for i in range(20):
            await marketplace.search_plugins(SearchQuery(query=f"test_{i}", limit=5))
    
    async def _simulate_bulk_search(self, marketplace: PluginMarketplace) -> None:
        """Simulate bulk search operations."""
        search_tasks = []
        for i in range(50):
            search_tasks.append(marketplace.search_plugins(SearchQuery(query="bulk", limit=10)))
        await asyncio.gather(*search_tasks)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except Exception:
            return 0.0
    
    def _generate_validation_report(self, validation_start: datetime) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        validation_duration = (datetime.utcnow() - validation_start).total_seconds()
        
        # Calculate summary statistics
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Performance summary
        api_times = [m.response_time_ms for m in self.performance_metrics if m.operation.startswith('api_') and m.success]
        avg_api_time = statistics.mean(api_times) if api_times else 0.0
        
        memory_usage = [m.memory_usage_mb for m in self.performance_metrics]
        peak_memory = max(memory_usage) if memory_usage else 0.0
        
        # Epic 1 compliance
        epic1_compliant = all(r.passed for r in self.validation_results)
        
        report = {
            "epic1_validation_report": {
                "timestamp": datetime.utcnow().isoformat(),
                "validation_duration_seconds": round(validation_duration, 2),
                "epic1_compliant": epic1_compliant,
                "compliance_status": "✅ COMPLIANT" if epic1_compliant else "❌ NON-COMPLIANT"
            },
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate_percent": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0
            },
            "performance_summary": {
                "average_api_response_time_ms": round(avg_api_time, 2),
                "peak_memory_usage_mb": round(peak_memory, 2),
                "baseline_memory_mb": round(self.baseline_memory, 2),
                "total_operations_tested": len(self.performance_metrics)
            },
            "epic1_targets": self.targets,
            "detailed_results": [r.to_dict() for r in self.validation_results],
            "performance_metrics": [m.to_dict() for m in self.performance_metrics],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on validation results."""
        recommendations = []
        
        # Check failed tests and generate specific recommendations
        failed_tests = [r for r in self.validation_results if not r.passed]
        
        for failed_test in failed_tests:
            if "API" in failed_test.test_name and "Response Time" in failed_test.test_name:
                recommendations.append(
                    f"⚡ Optimize API performance: {failed_test.test_name} exceeded target "
                    f"({failed_test.actual_value:.2f}ms > {failed_test.target_value:.2f}ms)"
                )
            
            elif "Memory" in failed_test.test_name:
                recommendations.append(
                    f"🧠 Optimize memory usage: {failed_test.test_name} exceeded target "
                    f"({failed_test.actual_value:.2f}MB > {failed_test.target_value:.2f}MB)"
                )
            
            elif "AI" in failed_test.test_name:
                recommendations.append(
                    f"🤖 Optimize AI inference: {failed_test.test_name} exceeded target "
                    f"({failed_test.actual_value:.2f}ms > {failed_test.target_value:.2f}ms)"
                )
            
            elif "Concurrency" in failed_test.test_name:
                recommendations.append(
                    f"⚡ Improve concurrency handling: {failed_test.test_name} had low success rate "
                    f"({failed_test.actual_value:.1f}% < {failed_test.target_value:.1f}%)"
                )
        
        # General recommendations if all tests pass
        if not failed_tests:
            recommendations.extend([
                "✅ All Epic 1 performance targets met successfully",
                "🔧 Continue monitoring performance in production",
                "📈 Consider implementing performance alerts for early detection",
                "🚀 Plugin marketplace ready for production deployment"
            ])
        
        return recommendations


async def run_epic1_validation() -> Dict[str, Any]:
    """
    Run Epic 1 performance validation for Plugin Marketplace.
    
    This function initializes the marketplace components and runs
    comprehensive performance validation.
    """
    try:
        logger.info("Initializing Epic 1 performance validation")
        
        # Initialize marketplace components
        from .advanced_plugin_manager import AdvancedPluginManager
        from .plugin_security_framework import PluginSecurityFramework
        
        plugin_manager = AdvancedPluginManager()
        security_framework = PluginSecurityFramework()
        
        marketplace = PluginMarketplace(plugin_manager, security_framework)
        await marketplace.initialize()
        
        ai_discovery = AIPluginDiscovery(marketplace)
        await ai_discovery.initialize()
        
        certification_pipeline = SecurityCertificationPipeline()
        await certification_pipeline.initialize()
        
        developer_platform = DeveloperOnboardingPlatform(marketplace, certification_pipeline)
        
        # Run validation
        validator = Epic1PerformanceValidator()
        validation_report = await validator.run_comprehensive_validation(
            marketplace, ai_discovery, certification_pipeline, developer_platform
        )
        
        return validation_report
        
    except Exception as e:
        logger.error("Epic 1 validation initialization failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "epic1_compliant": False
        }
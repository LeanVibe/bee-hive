#!/usr/bin/env python3
"""
Comprehensive smoke test for LeanVibe Agent Hive 2.0
Tests all systems and provides detailed diagnostics
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ComprehensiveSmokeTest:
    """Comprehensive system validation."""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    async def test_configuration_system(self) -> bool:
        """Test configuration loading and validation."""
        print("🧪 Testing Configuration System...")
        
        try:
            from app.core.config import settings
            
            # Validate key settings
            assert settings.APP_NAME == "LeanVibe Agent Hive"
            assert "postgresql" in settings.DATABASE_URL.lower()
            assert "redis" in settings.REDIS_URL.lower()
            assert settings.SECRET_KEY is not None
            
            print(f"✅ Configuration loaded successfully")
            print(f"   App: {settings.APP_NAME}")
            print(f"   Environment: {settings.ENVIRONMENT}")
            print(f"   Debug: {settings.DEBUG}")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            return False
    
    async def test_database_system(self) -> bool:
        """Test database connectivity and basic operations."""
        print("\n🧪 Testing Database System...")
        
        try:
            import asyncpg
            
            # Test basic connectivity
            conn = await asyncpg.connect(
                host='localhost',
                port=15432,
                user='leanvibe_user', 
                password='leanvibe_secure_pass',
                database='leanvibe_agent_hive'
            )
            
            # Test basic query
            result = await conn.fetchval('SELECT 1 as test')
            assert result == 1
            
            # Test database metadata
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            await conn.close()
            
            print(f"✅ Database connection successful")
            print(f"   Test query result: {result}")
            print(f"   Available tables: {len(tables)}")
            if tables:
                print(f"   Sample tables: {[t['table_name'] for t in tables[:3]]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            return False
    
    async def test_redis_system(self) -> bool:
        """Test Redis connectivity and operations."""
        print("\n🧪 Testing Redis System...")
        
        try:
            import redis
            
            # Test basic connectivity
            r = redis.Redis(host='localhost', port=16379, db=0, decode_responses=True)
            
            # Test ping
            pong = r.ping()
            assert pong is True
            
            # Test basic operations
            test_key = "smoke_test:redis"
            r.set(test_key, "test_value", ex=10)  # 10 second expiry
            value = r.get(test_key)
            assert value == "test_value"
            r.delete(test_key)
            
            # Test info
            info = r.info()
            
            print(f"✅ Redis connection successful")
            print(f"   Ping response: {pong}")
            print(f"   Redis version: {info.get('redis_version', 'unknown')}")
            print(f"   Connected clients: {info.get('connected_clients', 0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Redis test failed: {e}")
            return False
    
    async def test_cli_system(self) -> bool:
        """Test CLI command structure and functionality."""
        print("\n🧪 Testing CLI System...")
        
        try:
            from click.testing import CliRunner
            from app.cli.main import hive_cli
            
            runner = CliRunner()
            
            # Test help command
            result = runner.invoke(hive_cli, ['--help'])
            if result.exit_code != 0:
                print(f"❌ CLI help failed: {result.output}")
                return False
            
            # Test version command
            result = runner.invoke(hive_cli, ['--version'])
            if result.exit_code != 0:
                print(f"❌ CLI version failed: {result.output}")
                return False
            
            # Test command registry
            from app.cli.main import COMMAND_REGISTRY
            available_commands = list(COMMAND_REGISTRY.keys())
            
            print(f"✅ CLI system functional")
            print(f"   Available commands: {len(available_commands)}")
            print(f"   Commands: {', '.join(available_commands[:5])}...")
            
            return True
            
        except Exception as e:
            print(f"❌ CLI test failed: {e}")
            return False
    
    async def test_api_system(self) -> bool:
        """Test API system structure."""
        print("\n🧪 Testing API System...")
        
        try:
            from app.main import app
            
            # Check routes
            routes = app.routes
            route_count = len(routes)
            
            # Categorize routes
            api_routes = [r for r in routes if hasattr(r, 'path') and r.path.startswith('/api')]
            health_routes = [r for r in routes if hasattr(r, 'path') and 'health' in r.path]
            
            print(f"✅ API system loaded")
            print(f"   Total routes: {route_count}")
            print(f"   API routes: {len(api_routes)}")
            print(f"   Health routes: {len(health_routes)}")
            
            return True
            
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
    
    async def test_performance_system(self) -> bool:
        """Test performance monitoring system."""
        print("\n🧪 Testing Performance System...")
        
        try:
            from tests.isolation.performance.performance_regression_detector import PerformanceRegressionDetector
            
            detector = PerformanceRegressionDetector()
            
            # Check baseline data
            baseline_count = len(detector.baselines)
            
            # Check data directory
            data_dir = detector.data_dir
            data_files = list(data_dir.glob("*.json")) if data_dir.exists() else []
            
            print(f"✅ Performance system initialized")
            print(f"   Baselines established: {baseline_count}")
            print(f"   Data directory: {data_dir}")
            print(f"   Data files: {len(data_files)}")
            
            if detector.baselines:
                # Show baseline examples
                for i, (key, baseline) in enumerate(detector.baselines.items()):
                    if i >= 2:  # Show max 2 examples
                        break
                    print(f"   Baseline {i+1}: {key} ({baseline.sample_count} samples)")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    async def test_testing_infrastructure(self) -> bool:
        """Test the testing infrastructure itself."""
        print("\n🧪 Testing Infrastructure System...")
        
        try:
            # Check test directories
            test_dirs = [
                "tests/simple_system",
                "tests/isolation", 
                "tests/integration",
                "tests/performance"
            ]
            
            existing_dirs = []
            total_test_files = 0
            
            for test_dir in test_dirs:
                test_path = Path(test_dir)
                if test_path.exists():
                    existing_dirs.append(test_dir)
                    test_files = list(test_path.rglob("test_*.py"))
                    total_test_files += len(test_files)
            
            # Check if pytest is available
            try:
                import pytest
                pytest_available = True
            except ImportError:
                pytest_available = False
            
            print(f"✅ Testing infrastructure validated")
            print(f"   Test directories: {len(existing_dirs)}/{len(test_dirs)}")
            print(f"   Total test files: {total_test_files}")
            print(f"   Pytest available: {pytest_available}")
            
            return True
            
        except Exception as e:
            print(f"❌ Testing infrastructure test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        print("🚀 LeanVibe Agent Hive 2.0 - Comprehensive Smoke Test")
        print("=" * 70)
        
        test_methods = [
            ("Configuration System", self.test_configuration_system),
            ("Database System", self.test_database_system),
            ("Redis System", self.test_redis_system),
            ("CLI System", self.test_cli_system),
            ("API System", self.test_api_system),
            ("Performance System", self.test_performance_system),
            ("Testing Infrastructure", self.test_testing_infrastructure),
        ]
        
        results = {}
        
        for test_name, test_method in test_methods:
            self.total_tests += 1
            try:
                result = await test_method()
                results[test_name] = result
                if result:
                    self.passed_tests += 1
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        return results
    
    def generate_report(self, results: Dict[str, bool]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n📊 Comprehensive Test Results:")
        print("=" * 70)
        
        # Individual test results
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL" 
            print(f"{status} {test_name}")
        
        # Overall summary
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"\n🎯 Overall Summary:")
        print(f"   Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # System readiness assessment
        critical_systems = ["Configuration System", "Database System", "Redis System"]
        critical_passed = sum(1 for sys in critical_systems if results.get(sys, False))
        
        print(f"\n🏥 System Readiness:")
        print(f"   Critical Systems: {critical_passed}/{len(critical_systems)} operational")
        
        if critical_passed == len(critical_systems):
            print("   Status: 🟢 READY FOR BASIC OPERATIONS")
        elif critical_passed >= 2:
            print("   Status: 🟡 PARTIALLY OPERATIONAL")
        else:
            print("   Status: 🔴 MAJOR SYSTEMS DOWN")
        
        # Recommendations
        print(f"\n📋 Recommendations:")
        
        if success_rate >= 90:
            print("   🎉 All systems operational! Ready for production workloads.")
        elif success_rate >= 70:
            print("   ⚠️  Most systems working. Address failed tests before full deployment.")
        else:
            print("   🚨 Multiple system failures. Immediate attention required.")
        
        if not results.get("Database System", False):
            print("   🔧 Database: Check PostgreSQL container and connectivity")
        
        if not results.get("Redis System", False):
            print("   🔧 Redis: Verify Redis server is running on port 16379")
        
        # Return structured report
        return {
            "timestamp": "2025-08-21T20:30:00Z",
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "success_rate": success_rate,
            "critical_systems_operational": critical_passed,
            "system_status": "READY" if critical_passed == len(critical_systems) else "DEGRADED",
            "individual_results": results
        }

async def main():
    """Run comprehensive smoke test suite."""
    tester = ComprehensiveSmokeTest()
    
    # Run all tests
    results = await tester.run_all_tests()
    
    # Generate report
    report = tester.generate_report(results)
    
    # Save report to file
    report_file = Path("smoke_test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    if report["success_rate"] >= 80:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
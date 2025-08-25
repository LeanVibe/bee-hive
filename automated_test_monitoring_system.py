#!/usr/bin/env python3
"""
🤖 LeanVibe Agent Hive 2.0 - Automated Test Monitoring System
==============================================================

Intelligent test monitoring and quality assurance automation system.
Provides continuous test validation, regression detection, and automated
quality gate enforcement for the 450+ test suite.

Generated: August 25, 2025
Purpose: Continuous quality assurance and test reliability monitoring
Integration: CLI, CI/CD pipeline, and qa-test-guardian subagent ready
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from time import time, sleep
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class TestExecutionMetrics:
    """Test execution metrics tracking"""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    success_rate: float
    execution_time: float
    performance_score: float
    quality_gate_passed: bool

@dataclass
class TestRegressionAlert:
    """Test regression detection and alerting"""
    test_name: str
    previous_state: str
    current_state: str
    regression_type: str
    severity: str
    timestamp: str

class AutomatedTestMonitor:
    """Automated test monitoring and quality assurance system"""
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history = []
        self.regression_alerts = []
        self.quality_thresholds = {
            'min_success_rate': 85.0,
            'max_execution_time': 300.0,  # 5 minutes
            'performance_regression_threshold': 0.1,  # 10% performance degradation
            'critical_test_failure_threshold': 0.05  # 5% critical test failures
        }
        
        # Setup monitoring directories
        self.monitoring_dir = Path("test_monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.monitoring_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.metrics_file = self.monitoring_dir / "test_metrics_history.json"
        self.alerts_file = self.monitoring_dir / "regression_alerts.json"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.monitoring_dir / "test_monitoring.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_historical_data(self):
        """Load historical test metrics and alerts"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = [TestExecutionMetrics(**item) for item in data]
        except Exception as e:
            self.logger.warning(f"Could not load historical metrics: {e}")
            
        try:
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                    self.regression_alerts = [TestRegressionAlert(**item) for item in data]
        except Exception as e:
            self.logger.warning(f"Could not load historical alerts: {e}")
    
    def save_metrics_and_alerts(self):
        """Save current metrics and alerts to persistent storage"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump([asdict(m) for m in self.metrics_history], f, indent=2)
                
            with open(self.alerts_file, 'w') as f:
                json.dump([asdict(a) for a in self.regression_alerts], f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save monitoring data: {e}")
    
    def execute_test_suite_sample(self, sample_size: int = 50) -> TestExecutionMetrics:
        """Execute a representative sample of the test suite for monitoring"""
        
        start_time = time()
        
        try:
            # Use the parallel test executor for consistent monitoring
            result = subprocess.run([
                sys.executable, '-c', f'''
import sys
sys.path.insert(0, ".")
from parallel_test_executor import ParallelTestExecutor

executor = ParallelTestExecutor(max_workers=8)
test_files = executor.discover_all_test_files()
sample_files = test_files[:{sample_size}]

results = {{
    "total": len(sample_files),
    "passed": 0,
    "failed": 0
}}

for test_file in sample_files:
    result = executor.execute_single_test(test_file)
    if result.success:
        results["passed"] += 1
    else:
        results["failed"] += 1

print(f"RESULTS:{{results['total']}}:{{results['passed']}}:{{results['failed']}}")
'''
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and "RESULTS:" in result.stdout:
                # Parse results
                results_line = [line for line in result.stdout.split('\n') if line.startswith('RESULTS:')][0]
                _, total, passed, failed = results_line.split(':')
                
                total, passed, failed = int(total), int(passed), int(failed)
                execution_time = time() - start_time
                success_rate = (passed / total * 100) if total > 0 else 0
                
                # Calculate performance score (higher is better)
                performance_score = min(100, (50 / execution_time) * 100) if execution_time > 0 else 0
                
                quality_gate_passed = (
                    success_rate >= self.quality_thresholds['min_success_rate'] and
                    execution_time <= self.quality_thresholds['max_execution_time']
                )
                
                return TestExecutionMetrics(
                    timestamp=datetime.now().isoformat(),
                    total_tests=total,
                    passed=passed,
                    failed=failed,
                    success_rate=success_rate,
                    execution_time=execution_time,
                    performance_score=performance_score,
                    quality_gate_passed=quality_gate_passed
                )
            else:
                raise Exception(f"Test execution failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute test suite sample: {e}")
            # Return failure metrics
            return TestExecutionMetrics(
                timestamp=datetime.now().isoformat(),
                total_tests=sample_size,
                passed=0,
                failed=sample_size,
                success_rate=0.0,
                execution_time=time() - start_time,
                performance_score=0.0,
                quality_gate_passed=False
            )
    
    def detect_regressions(self, current_metrics: TestExecutionMetrics) -> List[TestRegressionAlert]:
        """Detect test regressions compared to historical data"""
        alerts = []
        
        if len(self.metrics_history) == 0:
            return alerts
            
        # Get recent baseline (last 5 successful runs)
        recent_successful = [m for m in self.metrics_history[-10:] if m.quality_gate_passed]
        
        if not recent_successful:
            return alerts
            
        baseline = recent_successful[-1]  # Most recent successful run
        
        # Check for success rate regression
        if current_metrics.success_rate < baseline.success_rate - 5.0:  # 5% threshold
            alerts.append(TestRegressionAlert(
                test_name="Overall Test Suite",
                previous_state=f"{baseline.success_rate:.1f}% success rate",
                current_state=f"{current_metrics.success_rate:.1f}% success rate", 
                regression_type="Success Rate Decline",
                severity="HIGH" if current_metrics.success_rate < 75.0 else "MEDIUM",
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for performance regression
        perf_degradation = (current_metrics.execution_time - baseline.execution_time) / baseline.execution_time
        if perf_degradation > self.quality_thresholds['performance_regression_threshold']:
            alerts.append(TestRegressionAlert(
                test_name="Test Execution Performance",
                previous_state=f"{baseline.execution_time:.1f}s execution time",
                current_state=f"{current_metrics.execution_time:.1f}s execution time",
                regression_type="Performance Degradation", 
                severity="HIGH" if perf_degradation > 0.25 else "MEDIUM",
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for quality gate failure
        if not current_metrics.quality_gate_passed and baseline.quality_gate_passed:
            alerts.append(TestRegressionAlert(
                test_name="Quality Gate",
                previous_state="PASSING",
                current_state="FAILING",
                regression_type="Quality Gate Failure",
                severity="CRITICAL",
                timestamp=datetime.now().isoformat()
            ))
        
        return alerts
    
    def generate_monitoring_report(self, metrics: TestExecutionMetrics, alerts: List[TestRegressionAlert]) -> str:
        """Generate comprehensive monitoring report"""
        
        report = f"""
🤖 AUTOMATED TEST MONITORING REPORT
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 CURRENT TEST EXECUTION METRICS:
   Total Tests Sampled: {metrics.total_tests}
   Passed: {metrics.passed}
   Failed: {metrics.failed}
   Success Rate: {metrics.success_rate:.1f}%
   Execution Time: {metrics.execution_time:.2f}s
   Performance Score: {metrics.performance_score:.1f}/100
   Quality Gate: {'✅ PASSED' if metrics.quality_gate_passed else '❌ FAILED'}

🎯 QUALITY THRESHOLDS:
   Min Success Rate: {self.quality_thresholds['min_success_rate']}%
   Max Execution Time: {self.quality_thresholds['max_execution_time']}s
   Performance Regression Threshold: {self.quality_thresholds['performance_regression_threshold']*100}%

"""
        
        if alerts:
            report += f"""
🚨 REGRESSION ALERTS ({len(alerts)} detected):
"""
            for alert in alerts:
                severity_icon = "🔴" if alert.severity == "CRITICAL" else "🟡" if alert.severity == "HIGH" else "🟠"
                report += f"""
   {severity_icon} {alert.severity}: {alert.regression_type}
      Test: {alert.test_name}
      Previous: {alert.previous_state}
      Current: {alert.current_state}
      Time: {alert.timestamp}
"""
        else:
            report += "✅ No regressions detected\n"
            
        # Historical trend analysis
        if len(self.metrics_history) >= 2:
            recent_trends = self.analyze_trends()
            report += f"""
📈 HISTORICAL TRENDS (Last 10 runs):
   Average Success Rate: {recent_trends['avg_success_rate']:.1f}%
   Average Execution Time: {recent_trends['avg_execution_time']:.1f}s
   Success Rate Trend: {recent_trends['success_rate_trend']}
   Performance Trend: {recent_trends['performance_trend']}
   Quality Gate Pass Rate: {recent_trends['quality_gate_pass_rate']:.1f}%
"""
        
        report += f"""
🔄 MONITORING STATUS:
   Historical Data Points: {len(self.metrics_history)}
   Total Alerts Generated: {len(self.regression_alerts)}
   Monitoring Active: {'✅ YES' if self.monitoring_active else '❌ NO'}

📋 RECOMMENDATIONS:
"""
        
        # Generate recommendations based on current state
        recommendations = self.generate_recommendations(metrics, alerts)
        for rec in recommendations:
            report += f"   • {rec}\n"
            
        return report
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze historical trends in test metrics"""
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        if len(recent_metrics) < 2:
            return {}
            
        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        
        # Trend analysis
        first_half = recent_metrics[:len(recent_metrics)//2]
        second_half = recent_metrics[len(recent_metrics)//2:]
        
        success_rate_trend = "IMPROVING" if (
            sum(m.success_rate for m in second_half) / len(second_half) >
            sum(m.success_rate for m in first_half) / len(first_half)
        ) else "DECLINING"
        
        performance_trend = "IMPROVING" if (
            sum(m.execution_time for m in second_half) / len(second_half) <
            sum(m.execution_time for m in first_half) / len(first_half)
        ) else "DECLINING"
        
        quality_gate_pass_rate = sum(1 for m in recent_metrics if m.quality_gate_passed) / len(recent_metrics) * 100
        
        return {
            'avg_success_rate': avg_success_rate,
            'avg_execution_time': avg_execution_time,
            'success_rate_trend': success_rate_trend,
            'performance_trend': performance_trend,
            'quality_gate_pass_rate': quality_gate_pass_rate
        }
    
    def generate_recommendations(self, metrics: TestExecutionMetrics, alerts: List[TestRegressionAlert]) -> List[str]:
        """Generate actionable recommendations based on current metrics and alerts"""
        recommendations = []
        
        if not metrics.quality_gate_passed:
            recommendations.append("🚨 URGENT: Quality gate failed - investigate and fix failing tests immediately")
            
        if metrics.success_rate < 75.0:
            recommendations.append("🔧 CRITICAL: Success rate below 75% - review and fix failing test infrastructure")
            
        if metrics.execution_time > 240:  # 4 minutes
            recommendations.append("⚡ PERFORMANCE: Execution time approaching 5-minute limit - optimize test performance")
            
        if any(alert.severity == "CRITICAL" for alert in alerts):
            recommendations.append("🆘 ESCALATE: Critical regression detected - notify development team immediately")
            
        if len(alerts) > 3:
            recommendations.append("📊 INVESTIGATE: Multiple regressions detected - conduct comprehensive system review")
            
        if metrics.quality_gate_passed and metrics.success_rate > 90.0:
            recommendations.append("✅ EXCELLENT: System performing well - maintain current quality standards")
            recommendations.append("🚀 OPTIMIZE: Consider increasing test coverage or performance improvements")
            
        return recommendations
    
    def start_continuous_monitoring(self, interval_minutes: int = 30):
        """Start continuous test monitoring with specified interval"""
        
        self.monitoring_active = True
        self.load_historical_data()
        
        self.logger.info(f"🤖 Starting continuous test monitoring (interval: {interval_minutes} minutes)")
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self.logger.info("🧪 Executing test suite sample for monitoring...")
                    
                    # Execute test sample
                    current_metrics = self.execute_test_suite_sample(sample_size=25)  # Smaller sample for continuous monitoring
                    
                    # Detect regressions
                    new_alerts = self.detect_regressions(current_metrics)
                    
                    # Update history and alerts
                    self.metrics_history.append(current_metrics)
                    self.regression_alerts.extend(new_alerts)
                    
                    # Keep only last 100 metrics entries
                    if len(self.metrics_history) > 100:
                        self.metrics_history = self.metrics_history[-100:]
                    
                    # Generate and save report
                    report = self.generate_monitoring_report(current_metrics, new_alerts)
                    
                    # Save to file with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_file = self.reports_dir / f"monitoring_report_{timestamp}.txt"
                    with open(report_file, 'w') as f:
                        f.write(report)
                    
                    # Save persistent data
                    self.save_metrics_and_alerts()
                    
                    # Log summary
                    status = "✅ HEALTHY" if current_metrics.quality_gate_passed else "❌ ISSUES DETECTED"
                    self.logger.info(f"📊 Monitoring cycle complete: {status} | Success: {current_metrics.success_rate:.1f}% | Time: {current_metrics.execution_time:.1f}s")
                    
                    if new_alerts:
                        self.logger.warning(f"🚨 {len(new_alerts)} new regression alerts detected!")
                    
                    # Wait for next cycle
                    sleep(interval_minutes * 60)
                    
                except Exception as e:
                    self.logger.error(f"❌ Monitoring cycle failed: {e}")
                    sleep(60)  # Wait 1 minute before retry
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        return monitoring_thread
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        self.logger.info("🛑 Test monitoring stopped")
    
    def run_single_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a single monitoring cycle and return results"""
        
        self.load_historical_data()
        
        print("🤖 AUTOMATED TEST MONITORING SYSTEM - SINGLE CYCLE")
        print("=" * 55)
        
        # Execute test sample
        print("🧪 Executing test suite sample...")
        current_metrics = self.execute_test_suite_sample(sample_size=50)
        
        # Detect regressions
        print("🔍 Analyzing for regressions...")
        new_alerts = self.detect_regressions(current_metrics)
        
        # Update history
        self.metrics_history.append(current_metrics)
        self.regression_alerts.extend(new_alerts)
        
        # Generate report
        print("📊 Generating monitoring report...")
        report = self.generate_monitoring_report(current_metrics, new_alerts)
        
        # Print report
        print(report)
        
        # Save data
        self.save_metrics_and_alerts()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"monitoring_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_file}")
        
        return {
            "metrics": asdict(current_metrics),
            "alerts": [asdict(alert) for alert in new_alerts],
            "report_file": str(report_file),
            "quality_gate_passed": current_metrics.quality_gate_passed
        }

def main():
    """Execute automated test monitoring system"""
    
    monitor = AutomatedTestMonitor()
    
    # Run single monitoring cycle
    results = monitor.run_single_monitoring_cycle()
    
    return results

if __name__ == "__main__":
    results = main()
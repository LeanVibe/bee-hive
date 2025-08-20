#!/usr/bin/env python3
"""
Performance System Operations - Operational Management Script

Provides comprehensive operational management, validation, and monitoring
for the integrated LeanVibe Agent Hive 2.0 performance optimization and
monitoring system.

Usage:
    python performance_system_operations.py [command] [options]

Commands:
    start       - Start integrated performance system
    stop        - Stop performance system gracefully
    status      - Get system status and health
    validate    - Run performance validation tests
    report      - Generate comprehensive performance report
    optimize    - Trigger manual optimization cycle
    dashboard   - Manage Grafana dashboards
    alerts      - Check and manage alerts
    config      - Display or update configuration

Key Features:
- One-command system startup and management
- Comprehensive performance validation
- Real-time monitoring and health checks
- Automated operational procedures
- Integration with existing LeanVibe architecture
"""

import asyncio
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import subprocess

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from integration.performance_integration_manager import (
    PerformanceIntegrationManager, IntegrationConfiguration, create_integrated_performance_system
)
from core.universal_orchestrator import UniversalOrchestrator


class PerformanceSystemOperations:
    """Operational management for performance system."""
    
    def __init__(self):
        self.integration_manager: Optional[PerformanceIntegrationManager] = None
        self.config_file = Path(__file__).parent.parent / "config" / "performance_integration.json"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start_system(self, config_path: Optional[str] = None) -> bool:
        """Start integrated performance system."""
        try:
            self.logger.info("Starting LeanVibe Agent Hive 2.0 Performance System...")
            
            # Load configuration
            config = await self._load_configuration(config_path)
            
            # Initialize orchestrator (assuming it exists)
            orchestrator = await self._get_or_create_orchestrator()
            
            # Create integrated performance system
            self.integration_manager = await create_integrated_performance_system(
                orchestrator=orchestrator,
                config=config
            )
            
            # Verify system is operational
            await asyncio.sleep(10)  # Allow startup to complete
            status = await self.integration_manager.get_integration_status()
            
            if status['status'] == 'active':
                self.logger.info("✅ Performance system started successfully")
                self._print_startup_summary(status)
                return True
            else:
                self.logger.error(f"❌ Performance system startup failed: {status['status']}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Failed to start performance system: {e}")
            return False
    
    async def stop_system(self) -> bool:
        """Stop performance system gracefully."""
        try:
            self.logger.info("Stopping LeanVibe Agent Hive 2.0 Performance System...")
            
            if self.integration_manager:
                await self.integration_manager.shutdown()
                self.logger.info("✅ Performance system stopped gracefully")
                return True
            else:
                self.logger.warning("No active performance system to stop")
                return True
        
        except Exception as e:
            self.logger.error(f"❌ Failed to stop performance system: {e}")
            return False
    
    async def get_system_status(self, detailed: bool = False) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.integration_manager:
            return {'status': 'not_running', 'message': 'Performance system not initialized'}
        
        try:
            if detailed:
                return await self.integration_manager.get_comprehensive_report()
            else:
                return await self.integration_manager.get_integration_status()
        
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def validate_performance(self) -> bool:
        """Run comprehensive performance validation."""
        self.logger.info("Running performance validation tests...")
        
        if not self.integration_manager:
            self.logger.error("❌ Performance system not running")
            return False
        
        try:
            # Get current status
            status = await self.integration_manager.get_integration_status()
            
            # Validate system health
            validation_results = {
                'system_active': status['status'] == 'active',
                'components_healthy': status['health_summary']['components_active'] >= 6,
                'performance_targets_met': status['health_summary']['performance_targets_met'],
                'no_critical_issues': status['health_summary']['active_issues_count'] == 0
            }
            
            # Validate performance metrics
            current_performance = status['current_performance']
            targets = status['performance_targets']
            
            performance_validation = {}
            
            for metric, target in targets.items():
                if metric in current_performance:
                    current = current_performance[metric]
                    
                    if metric in ['task_assignment_latency_ms', 'memory_usage_mb', 'error_rate_percent']:
                        # Lower is better
                        performance_validation[metric] = {
                            'current': current,
                            'target': target,
                            'meets_target': current <= target,
                            'performance_ratio': current / target if target > 0 else float('inf')
                        }
                    else:
                        # Higher is better
                        performance_validation[metric] = {
                            'current': current,
                            'target': target,
                            'meets_target': current >= target,
                            'performance_ratio': current / target if target > 0 else 0
                        }
                else:
                    performance_validation[metric] = {
                        'current': None,
                        'target': target,
                        'meets_target': False,
                        'performance_ratio': 0
                    }
            
            # Calculate overall validation score
            total_checks = len(validation_results) + len(performance_validation)
            passed_checks = (
                sum(validation_results.values()) + 
                sum(1 for v in performance_validation.values() if v['meets_target'])
            )
            
            validation_score = passed_checks / total_checks
            validation_success = validation_score >= 0.8  # 80% threshold
            
            # Print validation results
            self._print_validation_results(
                validation_results, performance_validation, validation_score, validation_success
            )
            
            return validation_success
        
        except Exception as e:
            self.logger.error(f"❌ Performance validation failed: {e}")
            return False
    
    async def trigger_optimization(self) -> bool:
        """Trigger manual optimization cycle."""
        if not self.integration_manager:
            self.logger.error("❌ Performance system not running")
            return False
        
        try:
            self.logger.info("Triggering manual optimization cycle...")
            
            if self.integration_manager.tuning_engine:
                cycle = await self.integration_manager.tuning_engine._run_optimization_cycle()
                
                self.logger.info(f"✅ Optimization cycle completed:")
                self.logger.info(f"   Success: {cycle.cycle_success}")
                self.logger.info(f"   Improvement: {cycle.total_improvement_percent:.2f}%")
                self.logger.info(f"   Actions attempted: {len(cycle.actions_attempted)}")
                self.logger.info(f"   Actions successful: {len(cycle.actions_successful)}")
                
                return cycle.cycle_success
            else:
                self.logger.error("❌ Automated tuning engine not available")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Manual optimization failed: {e}")
            return False
    
    async def manage_dashboards(self, action: str) -> bool:
        """Manage Grafana dashboards."""
        if not self.integration_manager:
            self.logger.error("❌ Performance system not running")
            return False
        
        try:
            if not self.integration_manager.dashboard_manager:
                self.logger.error("❌ Dashboard manager not available")
                return False
            
            if action == 'create':
                self.logger.info("Creating Grafana dashboards...")
                await self.integration_manager.dashboard_manager.create_all_dashboards()
                self.logger.info("✅ Dashboards created successfully")
                return True
            
            elif action == 'status':
                status = await self.integration_manager.dashboard_manager.get_dashboard_status()
                self.logger.info(f"Dashboard Status: {json.dumps(status, indent=2)}")
                return True
            
            elif action == 'refresh':
                self.logger.info("Refreshing dashboard data...")
                await self.integration_manager.dashboard_manager.refresh_dashboard_data()
                self.logger.info("✅ Dashboard data refreshed")
                return True
            
            else:
                self.logger.error(f"❌ Unknown dashboard action: {action}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Dashboard management failed: {e}")
            return False
    
    async def check_alerts(self) -> Dict[str, Any]:
        """Check current alerts and system health."""
        if not self.integration_manager:
            return {'error': 'Performance system not running'}
        
        try:
            alert_info = {}
            
            if self.integration_manager.alerting_system:
                alert_status = await self.integration_manager.alerting_system.get_system_status()
                alert_info['alerting_system'] = alert_status
            
            # Get recent health issues
            status = await self.integration_manager.get_integration_status()
            alert_info['health_issues'] = {
                'active_issues_count': status['health_summary']['active_issues_count'],
                'components_active': status['health_summary']['components_active'],
                'performance_targets_met': status['health_summary']['performance_targets_met']
            }
            
            return alert_info
        
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
            return {'error': str(e)}
    
    async def generate_report(self, output_file: Optional[str] = None) -> bool:
        """Generate comprehensive performance report."""
        if not self.integration_manager:
            self.logger.error("❌ Performance system not running")
            return False
        
        try:
            self.logger.info("Generating comprehensive performance report...")
            
            report = await self.integration_manager.get_comprehensive_report()
            
            # Add timestamp and additional metadata
            report['report_metadata'] = {
                'generated_at': datetime.utcnow().isoformat(),
                'report_version': '1.0',
                'system_version': 'LeanVibe Agent Hive 2.0'
            }
            
            # Determine output location
            if output_file:
                output_path = Path(output_file)
            else:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                output_path = Path(__file__).parent.parent / "reports" / f"performance_report_{timestamp}.json"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"✅ Performance report generated: {output_path}")
            
            # Print summary
            self._print_report_summary(report)
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            return False
    
    def _print_startup_summary(self, status: Dict[str, Any]) -> None:
        """Print startup summary."""
        print("\n" + "="*80)
        print("🚀 LEANVIBE AGENT HIVE 2.0 PERFORMANCE SYSTEM")
        print("="*80)
        print(f"Status: {status['status'].upper()}")
        print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"Components Active: {status['health_summary']['components_active']}/{status['health_summary']['total_components']}")
        print(f"Performance Targets Met: {'✅' if status['health_summary']['performance_targets_met'] else '⚠️'}")
        
        print(f"\n📊 Current Performance:")
        for metric, value in status['current_performance'].items():
            target = status['performance_targets'].get(metric, 'N/A')
            print(f"  {metric}: {value} (target: {target})")
        
        print(f"\n🔧 Enabled Components:")
        for component, enabled in status['configuration'].items():
            print(f"  {component.replace('_', ' ').title()}: {'✅' if enabled else '❌'}")
        
        print("="*80 + "\n")
    
    def _print_validation_results(self, validation_results: Dict[str, bool], 
                                performance_validation: Dict[str, Dict[str, Any]],
                                validation_score: float, validation_success: bool) -> None:
        """Print validation results."""
        print("\n" + "="*80)
        print("🔍 PERFORMANCE VALIDATION RESULTS")
        print("="*80)
        
        print("System Health Checks:")
        for check, result in validation_results.items():
            print(f"  {check.replace('_', ' ').title()}: {'✅' if result else '❌'}")
        
        print("\nPerformance Metrics:")
        for metric, validation in performance_validation.items():
            status = '✅' if validation['meets_target'] else '❌'
            ratio = validation['performance_ratio']
            print(f"  {metric}: {status} ({validation['current']} vs {validation['target']}, ratio: {ratio:.2f})")
        
        print(f"\nOverall Validation Score: {validation_score:.1%}")
        print(f"Validation Result: {'✅ PASSED' if validation_success else '❌ FAILED'}")
        print("="*80 + "\n")
    
    def _print_report_summary(self, report: Dict[str, Any]) -> None:
        """Print report summary."""
        integration_status = report['integration_status']
        
        print("\n" + "="*80)
        print("📈 PERFORMANCE REPORT SUMMARY")
        print("="*80)
        print(f"System Status: {integration_status['status'].upper()}")
        print(f"Uptime: {integration_status['uptime_seconds']:.1f} seconds")
        print(f"Components: {integration_status['health_summary']['components_active']}/{integration_status['health_summary']['total_components']} active")
        
        # Show recent performance
        current_perf = integration_status['current_performance']
        if current_perf:
            print("\n📊 Current Performance:")
            for metric, value in current_perf.items():
                print(f"  {metric}: {value}")
        
        # Show health history if available
        health_summary = report.get('health_history_summary', {})
        if health_summary:
            print(f"\n🏥 Health Summary:")
            print(f"  Total Health Checks: {health_summary.get('total_health_checks', 0)}")
            print(f"  Average Components Active: {health_summary.get('average_components_active', 0):.1f}")
        
        print("="*80 + "\n")
    
    async def _load_configuration(self, config_path: Optional[str] = None) -> IntegrationConfiguration:
        """Load integration configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            # Convert to IntegrationConfiguration object
            return IntegrationConfiguration(**config_data)
        else:
            # Use default configuration
            return IntegrationConfiguration()
    
    async def _get_or_create_orchestrator(self) -> UniversalOrchestrator:
        """Get existing orchestrator or create new one."""
        # This would typically connect to existing orchestrator
        # For now, create a minimal instance
        orchestrator = UniversalOrchestrator()
        await orchestrator.initialize()
        return orchestrator


async def main():
    """Main entry point for operational script."""
    parser = argparse.ArgumentParser(
        description="LeanVibe Agent Hive 2.0 Performance System Operations"
    )
    
    parser.add_argument('command', choices=[
        'start', 'stop', 'status', 'validate', 'report', 'optimize', 'dashboard', 'alerts', 'config'
    ], help='Operation to perform')
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, help='Output file for reports')
    parser.add_argument('--detailed', action='store_true', help='Show detailed status')
    parser.add_argument('--dashboard-action', type=str, choices=['create', 'status', 'refresh'],
                       default='status', help='Dashboard management action')
    
    args = parser.parse_args()
    
    operations = PerformanceSystemOperations()
    
    try:
        if args.command == 'start':
            success = await operations.start_system(args.config)
            sys.exit(0 if success else 1)
        
        elif args.command == 'stop':
            success = await operations.stop_system()
            sys.exit(0 if success else 1)
        
        elif args.command == 'status':
            status = await operations.get_system_status(args.detailed)
            print(json.dumps(status, indent=2, default=str))
        
        elif args.command == 'validate':
            success = await operations.validate_performance()
            sys.exit(0 if success else 1)
        
        elif args.command == 'report':
            success = await operations.generate_report(args.output)
            sys.exit(0 if success else 1)
        
        elif args.command == 'optimize':
            success = await operations.trigger_optimization()
            sys.exit(0 if success else 1)
        
        elif args.command == 'dashboard':
            success = await operations.manage_dashboards(args.dashboard_action)
            sys.exit(0 if success else 1)
        
        elif args.command == 'alerts':
            alerts = await operations.check_alerts()
            print(json.dumps(alerts, indent=2, default=str))
        
        elif args.command == 'config':
            config = IntegrationConfiguration()
            print(json.dumps(config.__dict__, indent=2, default=str))
    
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        if operations.integration_manager:
            await operations.stop_system()
        sys.exit(130)
    
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class PerformanceSystemOperationsScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(PerformanceSystemOperationsScript)
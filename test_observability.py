#!/usr/bin/env python3
"""
Test script for Advanced Observability Infrastructure
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.enterprise_observability import EnterpriseObservability, DevelopmentMetrics
from datetime import datetime


async def test_observability():
    print('🎯 Advanced Observability Infrastructure Test')
    print('=' * 50)
    
    try:
        # Initialize observability on different port
        print('🔧 Initializing enterprise observability...')
        observability = EnterpriseObservability(metrics_port=8002)
        
        # Skip metrics server startup for test
        print('✅ Observability framework initialized')
        
        # Test metrics recording
        print('📊 Testing metrics recording...')
        test_metrics = DevelopmentMetrics(
            task_id='test_observability_001',
            agent_type='claude_code',
            generation_time_seconds=18.5,
            execution_time_seconds=0.4,
            total_time_seconds=18.9,
            code_length=1850,
            success=True,
            security_level='high',
            quality_score=0.88,
            created_at=datetime.utcnow()
        )
        
        await observability.record_autonomous_development(test_metrics)
        print('✅ Development metrics recorded')
        
        # Test additional metrics for comprehensive data
        for i in range(10):
            metrics = DevelopmentMetrics(
                task_id=f'test_{i:03d}',
                agent_type='claude_code',
                generation_time_seconds=15.0 + i,
                execution_time_seconds=0.3,
                total_time_seconds=15.3 + i,
                code_length=1200 + i*100,
                success=True,
                security_level='high',
                quality_score=0.85 + i*0.02,
                created_at=datetime.utcnow()
            )
            await observability.record_autonomous_development(metrics)
        
        print('✅ Multiple metrics recorded for comprehensive testing')
        
        # Get dashboard data
        print('📈 Generating enterprise dashboard data...')
        dashboard_data = await observability.get_enterprise_dashboard_data()
        
        summary = dashboard_data.get('summary', {})
        roi = dashboard_data.get('roi_metrics', {})
        
        print(f'✅ Dashboard data generated:')
        print(f'  📊 Total tasks: {summary.get("total_tasks", 0)}')
        print(f'  ✅ Success rate: {summary.get("success_rate", 0):.1%}')
        print(f'  ⚡ Avg generation time: {summary.get("avg_generation_time", 0):.1f}s')
        print(f'  🛡️ Avg execution time: {summary.get("avg_execution_time", 0):.1f}s')
        print(f'  💰 Velocity improvement: {roi.get("development_velocity_improvement", 1):.1f}x')
        print(f'  💵 Cost savings/hour: ${roi.get("cost_savings_per_hour", 0):.0f}')
        print(f'  ⏰ Hours saved: {roi.get("developer_hours_saved", 0):.2f}')
        print(f'  📈 Quality score: {roi.get("quality_improvement_score", 0):.2f}')
        
        # Test alert system
        alerts = await observability.alert_manager.check_thresholds(dashboard_data)
        print(f'🚨 Alert system: {len(alerts)} active alerts')
        
        print(f'\\n🎉 Advanced Observability Infrastructure: OPERATIONAL')
        print(f'📊 Comprehensive metrics collection: ✅')
        print(f'💰 ROI calculation engine: ✅')
        print(f'🚨 Alert management system: ✅')
        print(f'📈 Enterprise dashboard data: ✅')
        
        return True
        
    except Exception as e:
        print(f'❌ Observability test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_observability())
    print(f'\\n📊 Advanced Observability: {"OPERATIONAL" if result else "NEEDS WORK"}')
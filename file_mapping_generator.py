#!/usr/bin/env python3
"""
File Mapping Matrix Generator
Creates detailed mapping from current 313 files to target 50 modules
"""

import csv
from pathlib import Path

# Define the consolidation mapping
CONSOLIDATION_MAPPING = {
    # Core Orchestration (5 modules)
    'production_orchestrator.py': {
        'sources': [
            'orchestrator.py',
            'production_orchestrator.py', 
            'unified_production_orchestrator.py',
            'automated_orchestrator.py',
            'performance_orchestrator.py',
            'high_concurrency_orchestrator.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['database.py', 'redis.py', 'agent_registry.py']
    },
    
    'agent_lifecycle_manager.py': {
        'sources': [
            'agent_lifecycle_manager.py',
            'agent_lifecycle_hooks.py',
            'agent_spawner.py',
            'agent_registry.py',
            'agent_load_balancer.py',
            'agent_persona_system.py',
            'agent_identity_service.py',
            'capability_matcher.py'
        ],
        'type': 'merge',
        'risk': 'high',
        'dependencies': ['database.py', 'communication.py']
    },
    
    'task_execution_engine.py': {
        'sources': [
            'task_execution_engine.py',
            'task_scheduler.py',
            'task_queue.py',
            'task_distributor.py',
            'task_batch_executor.py',
            'intelligent_task_router.py',
            'enhanced_intelligent_task_router.py',
            'smart_scheduler.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['agent_lifecycle_manager.py', 'redis.py']
    },
    
    'coordination_hub.py': {
        'sources': [
            'coordination.py',
            'coordination_dashboard.py',
            'enhanced_multi_agent_coordination.py',
            'realtime_coordination_sync.py',
            'global_coordination_integration.py',
            'multi_agent_commands.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['messaging_service.py', 'websocket_manager.py']
    },
    
    'workflow_engine.py': {
        'sources': [
            'workflow_engine.py',
            'enhanced_workflow_engine.py',
            'workflow_intelligence.py',
            'workflow_state_manager.py',
            'workflow_context_manager.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['task_execution_engine.py']
    },
    
    # Communication & Messaging (8 modules)
    'messaging_service.py': {
        'sources': [
            'agent_communication_service.py',
            'agent_messaging_service.py',
            'communication.py',
            'message_processor.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['redis.py']
    },
    
    'websocket_manager.py': {
        'sources': [
            'realtime_coordination_sync.py',
            'realtime_dashboard_streaming.py',
            'transcript_streaming.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['messaging_service.py']
    },
    
    'redis_integration.py': {
        'sources': [
            'redis.py',
            'redis_pubsub_manager.py',
            'enhanced_redis_streams_manager.py',
            'optimized_redis.py',
            'team_coordination_redis.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['config.py']
    },
    
    'event_processing.py': {
        'sources': [
            'event_processor.py',
            'event_serialization.py',
            'workflow_message_router.py',
            'hook_processor.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['redis_integration.py']
    },
    
    'real_time_coordination.py': {
        'sources': [
            'realtime_coordination_sync.py',
            'realtime_dashboard_streaming.py',
            'observability_streams.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['websocket_manager.py', 'event_processing.py']
    },
    
    'load_balancing.py': {
        'sources': [
            'agent_load_balancer.py',
            'load_balancing_benchmarks.py',
            'distributed_load_balancing_state.py',
            'orchestrator_load_balancing_integration.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['agent_lifecycle_manager.py']
    },
    
    'circuit_breaker.py': {
        'sources': [
            'circuit_breaker.py',
            # Note: 8 files have CircuitBreaker class - consolidate into one
        ],
        'type': 'deduplicate',
        'risk': 'low',
        'dependencies': []
    },
    
    'communication_protocols.py': {
        'sources': [
            'communication_analyzer.py',
            'correlation.py',
            'stream_monitor.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['messaging_service.py']
    },
    
    # Security & Authentication (6 modules)
    'authentication_service.py': {
        'sources': [
            'auth.py',
            'oauth_authentication_system.py',
            'oauth_provider_system.py',
            'mfa_system.py',
            'webauthn_system.py',
            'enhanced_jwt_manager.py',
            'auth_metrics.py',
            'api_key_manager.py'
        ],
        'type': 'merge',
        'risk': 'high',
        'dependencies': ['database.py', 'secret_manager.py']
    },
    
    'authorization_engine.py': {
        'sources': [
            'authorization_engine.py',
            'rbac_engine.py',
            'access_control.py',
            'api_security_middleware.py',
            'security_validation_middleware.py',
            'production_api_security.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['authentication_service.py']
    },
    
    'security_monitoring.py': {
        'sources': [
            'security_monitoring_system.py',
            'security_audit.py',
            'enhanced_security_audit.py',
            'comprehensive_audit_system.py',
            'audit_logger.py',
            'security_middleware.py',
            'threat_detection_engine.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['logging_service.py']
    },
    
    'compliance_framework.py': {
        'sources': [
            'compliance_framework.py',
            'enterprise_compliance.py',
            'enterprise_compliance_system.py',
            'integrated_security_system.py',
            'advanced_security_validator.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['security_monitoring.py']
    },
    
    'encryption_service.py': {
        'sources': [
            'secret_manager.py',
            'enterprise_secrets_manager.py',
            'github_security.py',
            'secure_code_executor.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['config.py']
    },
    
    'threat_detection.py': {
        'sources': [
            'threat_detection_engine.py',
            'poison_message_detector.py',
            'enhanced_security_safeguards.py',
            'security_policy_engine.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['security_monitoring.py']
    },
    
    # Performance & Monitoring (7 modules)
    'performance_monitor.py': {
        'sources': [
            'performance_monitoring.py',
            'performance_evaluator.py',
            'performance_validator.py',
            'performance_metrics_collector.py',
            'performance_metrics_publisher.py',
            'performance_benchmarks.py',
            'vs_2_1_performance_validator.py',
            'database_performance_validator.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['metrics_collector.py']
    },
    
    'metrics_collector.py': {
        'sources': [
            'custom_metrics_exporter.py',
            'prometheus_exporter.py',
            'dashboard_metrics_streaming.py',
            'team_coordination_metrics.py',
            'context_performance_monitor.py',
            'performance_storage_engine.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['redis_integration.py']
    },
    
    'observability_engine.py': {
        'sources': [
            'observability_hooks.py',
            'observability_streams.py',
            'observability_performance_testing.py',
            'enterprise_observability.py',
            'enhanced_observability.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['performance_monitor.py']
    },
    
    'prometheus_integration.py': {
        'sources': [
            'prometheus_exporter.py',
            'enhanced_prometheus_integration.py',
            'dashboard_prometheus.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['metrics_collector.py']
    },
    
    'alert_manager.py': {
        'sources': [
            'intelligent_alerting.py',
            'alert_analysis_engine.py',
            'health_monitor.py',
            'dlq_monitoring.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['performance_monitor.py']
    },
    
    'performance_optimizer.py': {
        'sources': [
            'performance_optimizer.py',
            'performance_optimization_advisor.py',
            'performance_optimizations.py',
            'resource_optimizer.py',
            'gradient_optimizer.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['performance_monitor.py']
    },
    
    'benchmarking_framework.py': {
        'sources': [
            'load_testing.py',
            'orchestrator_load_testing.py',
            'enhanced_communication_load_testing.py',
            'hook_performance_benchmarks.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['performance_monitor.py']
    },
    
    # Context & Memory Management (6 modules)
    'context_engine.py': {
        'sources': [
            'context_manager.py',
            'advanced_context_engine.py',
            'enhanced_context_engine.py',
            'context_engine_integration.py',
            'context_aware_orchestrator_integration.py',
            'context_orchestrator_integration.py',
            'context_adapter.py',
            'context_analytics.py',
            'context_lifecycle_manager.py',
            'context_relevance_scorer.py',
            'workflow_context_manager.py',
            'enhanced_context_consolidator.py'
        ],
        'type': 'merge',
        'risk': 'high',
        'dependencies': ['memory_manager.py', 'vector_search.py']
    },
    
    'memory_manager.py': {
        'sources': [
            'enhanced_memory_manager.py',
            'context_memory_manager.py',
            'memory_hierarchy_manager.py',
            'memory_consolidation_service.py',
            'cross_agent_knowledge_manager.py',
            'context_cache_manager.py'
        ],
        'type': 'merge',
        'risk': 'high',
        'dependencies': ['database.py']
    },
    
    'context_compression.py': {
        'sources': [
            'context_compression.py',
            'context_compression_engine.py',
            'consolidation_engine.py',
            'context_consolidator.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['context_engine.py']
    },
    
    'semantic_memory.py': {
        'sources': [
            'semantic_memory_engine.py',
            'semantic_memory_integration.py',
            'semantic_memory_task_processor.py',
            'semantic_embedding_service.py',
            'semantic_integrity_validator.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['vector_search.py']
    },
    
    'knowledge_graph.py': {
        'sources': [
            'knowledge_graph_builder.py',
            'agent_knowledge_manager.py',
            'dependency_graph_builder.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['semantic_memory.py']
    },
    
    'vector_search.py': {
        'sources': [
            'vector_search.py',
            'vector_search_engine.py',
            'enhanced_vector_search.py',
            'advanced_vector_search.py',
            'memory_aware_vector_search.py',
            'hybrid_search_engine.py'
        ],
        'type': 'merge',
        'risk': 'medium',
        'dependencies': ['database.py']
    },
    
    # Continue with remaining modules...
    # External Integrations (8 modules)
    'github_integration.py': {
        'sources': [
            'enhanced_github_integration.py',
            'github_api_client.py',
            'github_quality_integration.py',
            'github_webhooks.py',
            'pull_request_automator.py'
        ],
        'type': 'merge',
        'risk': 'low',
        'dependencies': ['encryption_service.py']
    },
    
    # Infrastructure & Configuration (10 modules)
    'logging_service.py': {
        'sources': [
            # Consolidates 306 logger instances across all files
        ],
        'type': 'unify',
        'risk': 'low',
        'dependencies': ['config.py']
    }
}

def generate_file_mapping():
    """Generate comprehensive file mapping CSV"""
    
    # Get all files in core directory
    core_path = Path('/Users/bogdan/work/leanvibe-dev/bee-hive/app/core')
    all_files = [f.name for f in core_path.glob('*.py') if f.name not in ['__init__.py']]
    
    # Create mapping rows
    mapping_rows = []
    mapped_files = set()
    
    for target_module, info in CONSOLIDATION_MAPPING.items():
        for source_file in info['sources']:
            if source_file in all_files:
                mapping_rows.append({
                    'Current_File': source_file,
                    'Target_Module': target_module,
                    'Consolidation_Type': info['type'],
                    'Risk_Level': info['risk'],
                    'Dependencies': ', '.join(info['dependencies']),
                    'Phase': get_migration_phase(info['risk']),
                    'Priority': get_priority(info['type'], info['risk'])
                })
                mapped_files.add(source_file)
    
    # Add unmapped files
    unmapped_files = set(all_files) - mapped_files
    for file in unmapped_files:
        target = infer_target_module(file)
        mapping_rows.append({
            'Current_File': file,
            'Target_Module': target,
            'Consolidation_Type': 'inferred',
            'Risk_Level': 'unknown',
            'Dependencies': '',
            'Phase': '4',
            'Priority': 'low'
        })
    
    # Write CSV
    with open('/Users/bogdan/work/leanvibe-dev/bee-hive/file_mapping.csv', 'w', newline='') as csvfile:
        fieldnames = ['Current_File', 'Target_Module', 'Consolidation_Type', 'Risk_Level', 'Dependencies', 'Phase', 'Priority']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(mapping_rows, key=lambda x: (x['Phase'], x['Priority'], x['Current_File'])))
    
    print(f"📋 Generated file mapping for {len(mapping_rows)} files")
    print(f"   - Explicitly mapped: {len(mapped_files)}")
    print(f"   - Inferred mapping: {len(unmapped_files)}")
    
    # Generate statistics
    stats = {}
    for row in mapping_rows:
        phase = row['Phase']
        if phase not in stats:
            stats[phase] = 0
        stats[phase] += 1
    
    print("\n📊 Migration Phase Distribution:")
    for phase in sorted(stats.keys()):
        print(f"   Phase {phase}: {stats[phase]} files")

def get_migration_phase(risk_level):
    """Determine migration phase based on risk"""
    if risk_level == 'low':
        return '1'
    elif risk_level == 'medium':
        return '2'
    elif risk_level == 'high':
        return '3'
    else:
        return '4'

def get_priority(consolidation_type, risk_level):
    """Determine priority based on type and risk"""
    if consolidation_type == 'unify':
        return 'high'
    elif risk_level == 'low':
        return 'high'
    elif risk_level == 'medium':
        return 'medium'
    else:
        return 'low'

def infer_target_module(filename):
    """Infer target module for unmapped files"""
    name = filename.lower()
    
    if 'agent' in name:
        return 'agent_lifecycle_manager.py'
    elif 'orchestrator' in name:
        return 'production_orchestrator.py'
    elif 'security' in name or 'auth' in name:
        return 'authentication_service.py'
    elif 'performance' in name or 'monitoring' in name:
        return 'performance_monitor.py'
    elif 'context' in name or 'memory' in name:
        return 'context_engine.py'
    elif 'workflow' in name or 'task' in name:
        return 'workflow_engine.py'
    elif 'communication' in name or 'messaging' in name:
        return 'messaging_service.py'
    elif 'database' in name:
        return 'database_manager.py'
    elif 'redis' in name:
        return 'redis_integration.py'
    else:
        return 'infrastructure_utilities.py'

if __name__ == "__main__":
    generate_file_mapping()
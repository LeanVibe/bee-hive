"""
Production-Grade Multi-Tenant Architecture
Enterprise deployment system with complete tenant isolation and resource management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import secrets
from decimal import Decimal
import uuid
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, insert, text
from sqlalchemy.pool import QueuePool
import redis.asyncio as redis
from cryptography.fernet import Fernet
import jwt
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import aiohttp
from kubernetes import client, config as k8s_config
import docker

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.core.security import SecurityManager


class TenantTier(Enum):
    """Tenant service tiers."""
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"


class ResourceType(Enum):
    """Types of resources to manage."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    STORAGE_GB = "storage_gb"
    NETWORK_BANDWIDTH_MBPS = "network_bandwidth_mbps"
    DATABASE_CONNECTIONS = "database_connections"
    REDIS_CONNECTIONS = "redis_connections"
    CONCURRENT_AGENTS = "concurrent_agents"
    API_REQUESTS_PER_HOUR = "api_requests_per_hour"


class IsolationLevel(Enum):
    """Levels of tenant isolation."""
    SHARED_INFRASTRUCTURE = "shared_infrastructure"  # Basic: Shared resources with quotas
    DEDICATED_CONTAINERS = "dedicated_containers"    # Premium: Dedicated containers
    DEDICATED_NODES = "dedicated_nodes"              # Enterprise: Dedicated K8s nodes
    DEDICATED_CLUSTER = "dedicated_cluster"          # Enterprise+: Dedicated K8s cluster


@dataclass
class ResourceQuota:
    """Resource quota definition for a tenant."""
    resource_type: ResourceType
    allocated: float
    used: float
    limit: float
    unit: str
    reset_period: str  # hourly, daily, monthly
    last_reset: datetime


@dataclass
class TenantConfiguration:
    """Complete tenant configuration."""
    tenant_id: str
    tenant_name: str
    tier: TenantTier
    isolation_level: IsolationLevel
    resource_quotas: Dict[ResourceType, ResourceQuota]
    security_config: Dict[str, Any]
    network_config: Dict[str, Any]
    storage_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    backup_config: Dict[str, Any]
    compliance_requirements: List[str] = field(default_factory=list)
    custom_domains: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TenantDeployment:
    """Tenant deployment information."""
    deployment_id: str
    tenant_id: str
    cluster_name: str
    namespace: str
    service_endpoints: Dict[str, str]
    database_config: Dict[str, Any]
    redis_config: Dict[str, Any]
    ingress_config: Dict[str, Any]
    ssl_certificates: Dict[str, str]
    deployment_status: str
    health_status: str
    last_health_check: datetime


@dataclass
class TenantMetrics:
    """Real-time tenant metrics."""
    tenant_id: str
    timestamp: datetime
    resource_utilization: Dict[ResourceType, float]
    performance_metrics: Dict[str, float]
    availability_percentage: float
    error_rate: float
    response_time_p95: float
    active_users: int
    api_calls_per_hour: int
    cost_per_hour: Decimal


class TenantIsolationManager:
    """Manager for tenant isolation and resource allocation."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.k8s_client: Optional[client.ApiClient] = None
        self.docker_client: Optional[docker.DockerClient] = None
        
    async def initialize(self):
        """Initialize Kubernetes and Docker clients."""
        try:
            # Initialize Kubernetes client
            try:
                k8s_config.load_incluster_config()  # For in-cluster deployment
            except:
                k8s_config.load_kube_config()  # For local development
            
            self.k8s_client = client.ApiClient()
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            self.logger.info("Tenant isolation manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize isolation manager: {e}")
    
    async def create_tenant_infrastructure(self, tenant_config: TenantConfiguration) -> TenantDeployment:
        """Create isolated infrastructure for a new tenant."""
        
        self.logger.info(f"Creating infrastructure for tenant: {tenant_config.tenant_id}")
        
        deployment_id = f"deploy_{tenant_config.tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Create tenant namespace
            namespace = await self._create_tenant_namespace(tenant_config)
            
            # Setup network isolation
            network_config = await self._setup_network_isolation(tenant_config, namespace)
            
            # Create dedicated database resources
            database_config = await self._create_tenant_database(tenant_config, namespace)
            
            # Create dedicated Redis instance
            redis_config = await self._create_tenant_redis(tenant_config, namespace)
            
            # Deploy tenant-specific services
            service_endpoints = await self._deploy_tenant_services(tenant_config, namespace)
            
            # Setup ingress and SSL
            ingress_config = await self._setup_tenant_ingress(tenant_config, namespace)
            ssl_certificates = await self._provision_ssl_certificates(tenant_config)
            
            # Apply resource quotas
            await self._apply_resource_quotas(tenant_config, namespace)
            
            # Setup monitoring and logging
            await self._setup_tenant_monitoring(tenant_config, namespace)
            
            deployment = TenantDeployment(
                deployment_id=deployment_id,
                tenant_id=tenant_config.tenant_id,
                cluster_name=self._get_cluster_name(tenant_config),
                namespace=namespace,
                service_endpoints=service_endpoints,
                database_config=database_config,
                redis_config=redis_config,
                ingress_config=ingress_config,
                ssl_certificates=ssl_certificates,
                deployment_status="deployed",
                health_status="healthy",
                last_health_check=datetime.now()
            )
            
            # Store deployment information
            await self.redis.setex(
                f"tenant_deployment:{tenant_config.tenant_id}",
                86400 * 365,  # 1 year TTL
                json.dumps(deployment.__dict__, default=str)
            )
            
            return deployment
            
        except Exception as e:
            self.logger.error(f"Failed to create tenant infrastructure: {e}")
            # Cleanup partial deployment
            await self._cleanup_failed_deployment(tenant_config.tenant_id, deployment_id)
            raise
    
    async def _create_tenant_namespace(self, tenant_config: TenantConfiguration) -> str:
        """Create a dedicated Kubernetes namespace for the tenant."""
        
        namespace_name = f"tenant-{tenant_config.tenant_id}"
        
        # Create namespace manifest
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace_name,
                "labels": {
                    "tenant-id": tenant_config.tenant_id,
                    "tenant-tier": tenant_config.tier.value,
                    "isolation-level": tenant_config.isolation_level.value,
                    "managed-by": "leanvibe-agent-hive"
                },
                "annotations": {
                    "tenant.leanvibe.com/created-at": datetime.now().isoformat(),
                    "tenant.leanvibe.com/tier": tenant_config.tier.value
                }
            }
        }
        
        # Apply namespace
        v1 = client.CoreV1Api(self.k8s_client)
        try:
            v1.create_namespace(body=namespace_manifest)
            self.logger.info(f"Created namespace: {namespace_name}")
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Namespace already exists
                self.logger.info(f"Namespace {namespace_name} already exists")
            else:
                raise
        
        return namespace_name
    
    async def _setup_network_isolation(
        self, 
        tenant_config: TenantConfiguration, 
        namespace: str
    ) -> Dict[str, Any]:
        """Setup network isolation using Kubernetes Network Policies."""
        
        network_policy_manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"tenant-{tenant_config.tenant_id}-isolation",
                "namespace": namespace
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"namespaceSelector": {"matchLabels": {"name": namespace}}},
                            {"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}},
                            {"namespaceSelector": {"matchLabels": {"name": "monitoring"}}}
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {"namespaceSelector": {"matchLabels": {"name": namespace}}},
                            {"namespaceSelector": {"matchLabels": {"name": "kube-system"}}}
                        ]
                    },
                    {
                        # Allow external API calls
                        "to": [],
                        "ports": [
                            {"protocol": "TCP", "port": 443},
                            {"protocol": "TCP", "port": 80}
                        ]
                    }
                ]
            }
        }
        
        # Apply network policy
        networking_v1 = client.NetworkingV1Api(self.k8s_client)
        try:
            networking_v1.create_namespaced_network_policy(
                namespace=namespace,
                body=network_policy_manifest
            )
            self.logger.info(f"Created network policy for tenant: {tenant_config.tenant_id}")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                # Update existing policy
                networking_v1.patch_namespaced_network_policy(
                    name=f"tenant-{tenant_config.tenant_id}-isolation",
                    namespace=namespace,
                    body=network_policy_manifest
                )
        
        return {
            "network_policy": f"tenant-{tenant_config.tenant_id}-isolation",
            "isolation_level": tenant_config.isolation_level.value,
            "namespace": namespace
        }
    
    async def _create_tenant_database(
        self, 
        tenant_config: TenantConfiguration, 
        namespace: str
    ) -> Dict[str, Any]:
        """Create dedicated database resources for the tenant."""
        
        if tenant_config.isolation_level in [IsolationLevel.DEDICATED_NODES, IsolationLevel.DEDICATED_CLUSTER]:
            # Deploy dedicated PostgreSQL instance
            return await self._deploy_dedicated_postgresql(tenant_config, namespace)
        else:
            # Create separate database in shared PostgreSQL instance
            return await self._create_tenant_database_schema(tenant_config)
    
    async def _deploy_dedicated_postgresql(
        self, 
        tenant_config: TenantConfiguration, 
        namespace: str
    ) -> Dict[str, Any]:
        """Deploy a dedicated PostgreSQL instance for the tenant."""
        
        postgres_manifest = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": f"postgresql-{tenant_config.tenant_id}",
                "namespace": namespace
            },
            "spec": {
                "serviceName": f"postgresql-{tenant_config.tenant_id}",
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": f"postgresql-{tenant_config.tenant_id}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": f"postgresql-{tenant_config.tenant_id}"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "postgresql",
                            "image": "postgres:15-alpine",
                            "env": [
                                {"name": "POSTGRES_DB", "value": f"tenant_{tenant_config.tenant_id}"},
                                {"name": "POSTGRES_USER", "value": f"tenant_{tenant_config.tenant_id}"},
                                {"name": "POSTGRES_PASSWORD", "valueFrom": {
                                    "secretKeyRef": {
                                        "name": f"postgresql-{tenant_config.tenant_id}-secret",
                                        "key": "password"
                                    }
                                }}
                            ],
                            "ports": [{"containerPort": 5432}],
                            "volumeMounts": [{
                                "name": "postgresql-storage",
                                "mountPath": "/var/lib/postgresql/data"
                            }],
                            "resources": {
                                "requests": {
                                    "memory": f"{tenant_config.resource_quotas[ResourceType.MEMORY_GB].allocated * 0.3}Gi",
                                    "cpu": f"{tenant_config.resource_quotas[ResourceType.CPU_CORES].allocated * 0.2}"
                                },
                                "limits": {
                                    "memory": f"{tenant_config.resource_quotas[ResourceType.MEMORY_GB].limit * 0.3}Gi",
                                    "cpu": f"{tenant_config.resource_quotas[ResourceType.CPU_CORES].limit * 0.2}"
                                }
                            }
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": {
                        "name": "postgresql-storage"
                    },
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "resources": {
                            "requests": {
                                "storage": f"{tenant_config.resource_quotas[ResourceType.STORAGE_GB].allocated}Gi"
                            }
                        }
                    }
                }]
            }
        }
        
        # Create database password secret
        db_password = secrets.token_urlsafe(32)
        secret_manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"postgresql-{tenant_config.tenant_id}-secret",
                "namespace": namespace
            },
            "data": {
                "password": self._base64_encode(db_password)
            }
        }
        
        # Apply manifests
        apps_v1 = client.AppsV1Api(self.k8s_client)
        v1 = client.CoreV1Api(self.k8s_client)
        
        v1.create_namespaced_secret(namespace=namespace, body=secret_manifest)
        apps_v1.create_namespaced_stateful_set(namespace=namespace, body=postgres_manifest)
        
        # Create service
        service_manifest = {
            "apiVersion": "v1", 
            "kind": "Service",
            "metadata": {
                "name": f"postgresql-{tenant_config.tenant_id}",
                "namespace": namespace
            },
            "spec": {
                "selector": {
                    "app": f"postgresql-{tenant_config.tenant_id}"
                },
                "ports": [{
                    "port": 5432,
                    "targetPort": 5432
                }]
            }
        }
        
        v1.create_namespaced_service(namespace=namespace, body=service_manifest)
        
        return {
            "type": "dedicated",
            "host": f"postgresql-{tenant_config.tenant_id}.{namespace}.svc.cluster.local",
            "port": 5432,
            "database": f"tenant_{tenant_config.tenant_id}",
            "username": f"tenant_{tenant_config.tenant_id}",
            "password": db_password,
            "connection_pool_size": tenant_config.resource_quotas[ResourceType.DATABASE_CONNECTIONS].limit
        }
    
    async def _apply_resource_quotas(
        self, 
        tenant_config: TenantConfiguration, 
        namespace: str
    ):
        """Apply Kubernetes resource quotas to the tenant namespace."""
        
        # Calculate total resource quotas
        total_cpu = sum(
            quota.limit for resource_type, quota in tenant_config.resource_quotas.items()
            if resource_type == ResourceType.CPU_CORES
        )
        total_memory = sum(
            quota.limit for resource_type, quota in tenant_config.resource_quotas.items()
            if resource_type == ResourceType.MEMORY_GB
        )
        total_storage = sum(
            quota.limit for resource_type, quota in tenant_config.resource_quotas.items()
            if resource_type == ResourceType.STORAGE_GB
        )
        
        quota_manifest = {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": f"tenant-{tenant_config.tenant_id}-quota",
                "namespace": namespace
            },
            "spec": {
                "hard": {
                    "requests.cpu": f"{total_cpu * 0.8}",  # 80% for requests
                    "requests.memory": f"{total_memory * 0.8}Gi",
                    "limits.cpu": f"{total_cpu}",
                    "limits.memory": f"{total_memory}Gi",
                    "persistentvolumeclaims": "10",
                    "requests.storage": f"{total_storage}Gi",
                    "pods": "50",
                    "services": "20",
                    "secrets": "10"
                }
            }
        }
        
        # Apply resource quota
        v1 = client.CoreV1Api(self.k8s_client)
        try:
            v1.create_namespaced_resource_quota(namespace=namespace, body=quota_manifest)
            self.logger.info(f"Applied resource quota for tenant: {tenant_config.tenant_id}")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                v1.patch_namespaced_resource_quota(
                    name=f"tenant-{tenant_config.tenant_id}-quota",
                    namespace=namespace,
                    body=quota_manifest
                )


class TenantResourceManager:
    """Manager for tenant resource allocation and monitoring."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.resource_cache: Dict[str, TenantMetrics] = {}
    
    async def monitor_tenant_resources(self, tenant_id: str) -> TenantMetrics:
        """Monitor resource usage for a specific tenant."""
        
        try:
            # Get tenant deployment info
            deployment_data = await self.redis.get(f"tenant_deployment:{tenant_id}")
            if not deployment_data:
                raise ValueError(f"Tenant deployment {tenant_id} not found")
            
            deployment = TenantDeployment(**json.loads(deployment_data))
            
            # Collect resource metrics from Kubernetes
            k8s_metrics = await self._collect_kubernetes_metrics(deployment)
            
            # Collect application-level metrics
            app_metrics = await self._collect_application_metrics(deployment)
            
            # Collect cost metrics
            cost_metrics = await self._calculate_cost_metrics(deployment, k8s_metrics)
            
            metrics = TenantMetrics(
                tenant_id=tenant_id,
                timestamp=datetime.now(),
                resource_utilization=k8s_metrics["resource_utilization"],
                performance_metrics=app_metrics["performance"],
                availability_percentage=app_metrics["availability"],
                error_rate=app_metrics["error_rate"],
                response_time_p95=app_metrics["response_time_p95"],
                active_users=app_metrics["active_users"],
                api_calls_per_hour=app_metrics["api_calls_per_hour"],
                cost_per_hour=cost_metrics["cost_per_hour"]
            )
            
            # Cache metrics
            self.resource_cache[tenant_id] = metrics
            
            # Store metrics for historical analysis
            await self.redis.setex(
                f"tenant_metrics:{tenant_id}:{datetime.now().strftime('%Y%m%d_%H')}",
                86400,  # 24 hours TTL
                json.dumps(metrics.__dict__, default=str)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to monitor tenant resources: {e}")
            raise
    
    async def _collect_kubernetes_metrics(self, deployment: TenantDeployment) -> Dict[str, Any]:
        """Collect resource metrics from Kubernetes."""
        
        # This would integrate with Kubernetes metrics server
        # For demo purposes, returning mock data
        return {
            "resource_utilization": {
                ResourceType.CPU_CORES: 2.5,
                ResourceType.MEMORY_GB: 8.2,
                ResourceType.STORAGE_GB: 150.0,
                ResourceType.NETWORK_BANDWIDTH_MBPS: 100.0
            }
        }
    
    async def _collect_application_metrics(self, deployment: TenantDeployment) -> Dict[str, Any]:
        """Collect application-level metrics."""
        
        # This would collect metrics from application monitoring
        return {
            "performance": {
                "avg_response_time_ms": 45.0,
                "throughput_rps": 150.0,
                "cache_hit_rate": 0.85
            },
            "availability": 99.95,
            "error_rate": 0.05,
            "response_time_p95": 95.0,
            "active_users": 45,
            "api_calls_per_hour": 9000
        }


class MultiTenantArchitectureService:
    """Main service for managing multi-tenant architecture."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client: Optional[redis.Redis] = None
        self.isolation_manager: Optional[TenantIsolationManager] = None
        self.resource_manager: Optional[TenantResourceManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.tenants: Dict[str, TenantConfiguration] = {}
    
    async def initialize(self):
        """Initialize the multi-tenant architecture service."""
        self.redis_client = await get_redis_client()
        self.isolation_manager = TenantIsolationManager(self.redis_client, self.logger)
        self.resource_manager = TenantResourceManager(self.redis_client, self.logger)
        self.security_manager = SecurityManager()
        
        await self.isolation_manager.initialize()
        
        self.logger.info("Multi-tenant architecture service initialized successfully")
    
    async def create_tenant(
        self,
        tenant_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new tenant with complete isolation."""
        
        tenant_id = f"tenant_{tenant_request['organization_name'].lower().replace(' ', '_')}_{secrets.token_hex(4)}"
        
        try:        
            # Determine tier and isolation level
            tier = TenantTier(tenant_request.get("tier", "basic"))
            isolation_level = self._determine_isolation_level(tier)
            
            # Create resource quotas based on tier
            resource_quotas = self._create_resource_quotas(tier, tenant_request.get("custom_limits", {}))
            
            # Create tenant configuration
            tenant_config = TenantConfiguration(
                tenant_id=tenant_id,
                tenant_name=tenant_request["organization_name"],
                tier=tier,
                isolation_level=isolation_level,
                resource_quotas=resource_quotas,
                security_config=self._create_security_config(tenant_request),
                network_config=self._create_network_config(tenant_request),
                storage_config=self._create_storage_config(tenant_request),
                monitoring_config=self._create_monitoring_config(tenant_request),
                backup_config=self._create_backup_config(tenant_request),
                compliance_requirements=tenant_request.get("compliance_requirements", []),
                custom_domains=tenant_request.get("custom_domains", [])
            )
            
            # Create tenant infrastructure
            deployment = await self.isolation_manager.create_tenant_infrastructure(tenant_config)
            
            # Store tenant configuration
            self.tenants[tenant_id] = tenant_config
            await self.redis_client.setex(
                f"tenant_config:{tenant_id}",
                86400 * 365,  # 1 year TTL
                json.dumps(tenant_config.__dict__, default=str)
            )
            
            # Generate tenant access credentials
            access_credentials = await self._generate_tenant_credentials(tenant_config)
            
            return {
                "status": "success", 
                "tenant_id": tenant_id,
                "tenant_configuration": tenant_config.__dict__,
                "deployment_info": deployment.__dict__,
                "access_credentials": access_credentials,
                "service_endpoints": deployment.service_endpoints,
                "estimated_setup_time": "15-30 minutes",
                "next_steps": [
                    "Complete DNS configuration for custom domains",
                    "Upload SSL certificates (if not using auto-generation)",
                    "Configure SSO integration",
                    "Initial data migration (if applicable)"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create tenant: {e}")
            return {
                "status": "error",
                "tenant_id": tenant_id,
                "error_message": str(e)
            }
    
    def _determine_isolation_level(self, tier: TenantTier) -> IsolationLevel:
        """Determine isolation level based on tenant tier."""
        
        isolation_mapping = {
            TenantTier.BASIC: IsolationLevel.SHARED_INFRASTRUCTURE,
            TenantTier.PREMIUM: IsolationLevel.DEDICATED_CONTAINERS,
            TenantTier.ENTERPRISE: IsolationLevel.DEDICATED_NODES,
            TenantTier.ENTERPRISE_PLUS: IsolationLevel.DEDICATED_CLUSTER
        }
        
        return isolation_mapping[tier]
    
    def _create_resource_quotas(
        self, 
        tier: TenantTier, 
        custom_limits: Dict[str, float]
    ) -> Dict[ResourceType, ResourceQuota]:
        """Create resource quotas based on tenant tier."""
        
        # Base quotas per tier
        tier_quotas = {
            TenantTier.BASIC: {
                ResourceType.CPU_CORES: 4.0,
                ResourceType.MEMORY_GB: 16.0,
                ResourceType.STORAGE_GB: 100.0,
                ResourceType.NETWORK_BANDWIDTH_MBPS: 100.0,
                ResourceType.DATABASE_CONNECTIONS: 20,
                ResourceType.REDIS_CONNECTIONS: 50,
                ResourceType.CONCURRENT_AGENTS: 5,
                ResourceType.API_REQUESTS_PER_HOUR: 10000
            },
            TenantTier.PREMIUM: {
                ResourceType.CPU_CORES: 8.0,
                ResourceType.MEMORY_GB: 32.0,
                ResourceType.STORAGE_GB: 500.0,
                ResourceType.NETWORK_BANDWIDTH_MBPS: 500.0,
                ResourceType.DATABASE_CONNECTIONS: 50,
                ResourceType.REDIS_CONNECTIONS: 100,
                ResourceType.CONCURRENT_AGENTS: 15,
                ResourceType.API_REQUESTS_PER_HOUR: 50000
            },
            TenantTier.ENTERPRISE: {
                ResourceType.CPU_CORES: 16.0,
                ResourceType.MEMORY_GB: 64.0,
                ResourceType.STORAGE_GB: 2000.0,
                ResourceType.NETWORK_BANDWIDTH_MBPS: 1000.0,
                ResourceType.DATABASE_CONNECTIONS: 100,
                ResourceType.REDIS_CONNECTIONS: 200,
                ResourceType.CONCURRENT_AGENTS: 50,
                ResourceType.API_REQUESTS_PER_HOUR: 200000
            },
            TenantTier.ENTERPRISE_PLUS: {
                ResourceType.CPU_CORES: 32.0,
                ResourceType.MEMORY_GB: 128.0,
                ResourceType.STORAGE_GB: 5000.0,
                ResourceType.NETWORK_BANDWIDTH_MBPS: 2000.0,
                ResourceType.DATABASE_CONNECTIONS: 200,
                ResourceType.REDIS_CONNECTIONS: 500, 
                ResourceType.CONCURRENT_AGENTS: 100,
                ResourceType.API_REQUESTS_PER_HOUR: 1000000
            }
        }
        
        base_quotas = tier_quotas[tier]
        resource_quotas = {}
        
        for resource_type, base_limit in base_quotas.items():
            # Apply custom limits if provided
            limit = custom_limits.get(resource_type.value, base_limit)
            
            resource_quotas[resource_type] = ResourceQuota(
                resource_type=resource_type,
                allocated=limit * 0.8,  # 80% allocation initially
                used=0.0,
                limit=limit,
                unit=self._get_resource_unit(resource_type),
                reset_period="hourly" if "per_hour" in resource_type.value else "monthly",
                last_reset=datetime.now()
            )
        
        return resource_quotas
    
    def _get_resource_unit(self, resource_type: ResourceType) -> str:
        """Get the unit for a resource type."""
        
        unit_mapping = {
            ResourceType.CPU_CORES: "cores",
            ResourceType.MEMORY_GB: "GB",
            ResourceType.STORAGE_GB: "GB",
            ResourceType.NETWORK_BANDWIDTH_MBPS: "Mbps",
            ResourceType.DATABASE_CONNECTIONS: "connections",
            ResourceType.REDIS_CONNECTIONS: "connections",
            ResourceType.CONCURRENT_AGENTS: "agents",
            ResourceType.API_REQUESTS_PER_HOUR: "requests/hour"
        }
        
        return unit_mapping.get(resource_type, "units")


# Global service instance
_multi_tenant_service: Optional[MultiTenantArchitectureService] = None


async def get_multi_tenant_service() -> MultiTenantArchitectureService:
    """Get the global multi-tenant architecture service instance."""
    global _multi_tenant_service
    
    if _multi_tenant_service is None:
        _multi_tenant_service = MultiTenantArchitectureService()
        await _multi_tenant_service.initialize()
    
    return _multi_tenant_service


# Usage example and testing
if __name__ == "__main__":
    async def test_multi_tenant_architecture():
        """Test the multi-tenant architecture service."""
        
        service = await get_multi_tenant_service()
        
        # Sample tenant request
        tenant_request = {
            "organization_name": "TechCorp Enterprise",
            "tier": "enterprise",
            "compliance_requirements": ["SOC2", "GDPR", "HIPAA"],
            "custom_domains": ["api.techcorp.com", "app.techcorp.com"],
            "custom_limits": {
                "cpu_cores": 24.0,
                "memory_gb": 96.0,
                "concurrent_agents": 75
            },
            "security_requirements": {
                "sso_integration": True,
                "ip_whitelist": ["192.168.1.0/24", "10.0.0.0/8"],
                "encryption_at_rest": True,
                "audit_logging": True
            }
        }
        
        # Create tenant
        result = await service.create_tenant(tenant_request)
        print("Tenant creation result:", json.dumps(result, indent=2, default=str))
        
        if result["status"] == "success":
            tenant_id = result["tenant_id"]
            
            # Monitor tenant resources
            metrics = await service.resource_manager.monitor_tenant_resources(tenant_id)
            print("Tenant metrics:", json.dumps(metrics.__dict__, indent=2, default=str))
    
    # Run test
    asyncio.run(test_multi_tenant_architecture())
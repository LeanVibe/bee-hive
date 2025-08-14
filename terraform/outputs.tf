# LeanVibe Agent Hive 2.0 - Terraform Outputs
# Output values for infrastructure components

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnet_ids
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = module.vpc.nat_gateway_ids
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = module.vpc.internet_gateway_id
}

# EKS Outputs
output "cluster_id" {
  description = "Name/ID of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_arn" {
  description = "ARN of the EKS cluster"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_version" {
  description = "Kubernetes server version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

output "node_group_arns" {
  description = "ARNs of the EKS node groups"
  value       = module.eks.node_group_arns
}

output "node_group_status" {
  description = "Status of the EKS node groups"
  value       = module.eks.node_group_status
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.rds.port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.database_name
}

output "rds_username" {
  description = "RDS master username"
  value       = module.rds.username
  sensitive   = true
}

output "rds_instance_id" {
  description = "RDS instance ID"
  value       = module.rds.instance_id
}

output "rds_instance_class" {
  description = "RDS instance class"
  value       = module.rds.instance_class
}

output "rds_allocated_storage" {
  description = "RDS allocated storage"
  value       = module.rds.allocated_storage
}

output "rds_backup_retention_period" {
  description = "RDS backup retention period"
  value       = module.rds.backup_retention_period
}

output "rds_security_group_id" {
  description = "Security group ID for RDS"
  value       = aws_security_group.rds.id
}

# ElastiCache Outputs
output "elasticache_cluster_id" {
  description = "ElastiCache cluster identifier"
  value       = module.elasticache.cluster_id
}

output "elasticache_endpoint" {
  description = "ElastiCache primary endpoint"
  value       = module.elasticache.primary_endpoint
  sensitive   = true
}

output "elasticache_port" {
  description = "ElastiCache port"
  value       = module.elasticache.port
}

output "elasticache_parameter_group" {
  description = "ElastiCache parameter group"
  value       = module.elasticache.parameter_group_name
}

output "elasticache_security_group_id" {
  description = "Security group ID for ElastiCache"
  value       = aws_security_group.elasticache.id
}

# Security Group Outputs
output "security_groups" {
  description = "Security group IDs"
  value = {
    rds         = aws_security_group.rds.id
    elasticache = aws_security_group.elasticache.id
  }
}

# Monitoring Outputs
output "monitoring_enabled" {
  description = "Whether monitoring is enabled"
  value       = var.enable_monitoring
}

output "prometheus_endpoint" {
  description = "Prometheus endpoint URL"
  value       = var.enable_monitoring ? module.monitoring[0].prometheus_endpoint : null
}

output "grafana_endpoint" {
  description = "Grafana endpoint URL"
  value       = var.enable_monitoring ? module.monitoring[0].grafana_endpoint : null
  sensitive   = true
}

output "grafana_admin_password" {
  description = "Grafana admin password"
  value       = var.enable_monitoring ? random_password.grafana_admin.result : null
  sensitive   = true
}

# KMS Outputs
output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = var.environment == "production" ? aws_kms_key.eks[0].id : null
}

output "kms_key_arn" {
  description = "KMS key ARN for encryption"
  value       = var.environment == "production" ? aws_kms_key.eks[0].arn : null
}

# IAM Outputs
output "iam_roles" {
  description = "IAM role ARNs"
  value = {
    rds_enhanced_monitoring = aws_iam_role.rds_enhanced_monitoring.arn
  }
}

# DNS and Load Balancer Outputs
output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = var.domain_name != "" ? var.domain_name : "eks-cluster-endpoint"
}

# Application Configuration Outputs
output "application_config" {
  description = "Application configuration values"
  value = {
    environment     = var.environment
    app_version     = var.app_version
    max_agents      = var.max_agents
    app_replicas    = var.app_replicas
    cluster_name    = module.eks.cluster_name
    namespace       = "leanvibe-agent-hive-${var.environment}"
  }
}

# Connection Strings
output "database_connection_string" {
  description = "Database connection string (without password)"
  value       = "postgresql://${module.rds.username}:REDACTED@${module.rds.endpoint}:${module.rds.port}/${module.rds.database_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${module.elasticache.primary_endpoint}:${module.elasticache.port}/0"
  sensitive   = true
}

# Kubectl Configuration
output "kubectl_config" {
  description = "kubectl configuration command"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# Environment Summary
output "environment_summary" {
  description = "Summary of the deployed environment"
  value = {
    project     = var.project_name
    environment = var.environment
    region      = var.aws_region
    vpc_cidr    = var.vpc_cidr
    cluster     = module.eks.cluster_name
    database    = module.rds.instance_id
    cache       = module.elasticache.cluster_id
    monitoring  = var.enable_monitoring
    created_at  = timestamp()
  }
}

# Cost Estimation Tags
output "cost_tags" {
  description = "Tags for cost tracking and allocation"
  value = {
    Project     = var.project_name
    Environment = var.environment
    CostCenter  = var.cost_center
    Owner       = var.owner
    ManagedBy   = "terraform"
  }
}

# Security Configuration
output "security_config" {
  description = "Security configuration summary"
  value = {
    encryption_at_rest    = var.enable_encryption_at_rest
    encryption_in_transit = var.enable_encryption_in_transit
    cluster_encryption    = var.environment == "production"
    rds_encryption        = true
    elasticache_encryption = true
  }
  sensitive = true
}

# Backup Configuration
output "backup_config" {
  description = "Backup configuration summary"
  value = {
    rds_backup_retention    = module.rds.backup_retention_period
    rds_backup_window       = "03:00-04:00"
    elasticache_snapshots   = var.environment == "production" ? 7 : 1
    automated_backups       = true
  }
}
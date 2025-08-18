# Terraform Variables for LeanVibe Agent Hive 2.0 Production Deployment

# Basic Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "leanvibe-agent-hive"
}

variable "environment" {
  description = "Deployment environment (production, staging, development)"
  type        = string
  default     = "production"
}

variable "project_owner" {
  description = "Project owner/team"
  type        = string
  default     = "leanvibe-team"
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

# Terraform State Configuration
variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state storage"
  type        = string
}

variable "terraform_lock_table" {
  description = "DynamoDB table for Terraform state locking"
  type        = string
}

# EKS Configuration
variable "eks_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_group_instance_types" {
  description = "EC2 instance types for EKS node groups"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the EKS node group"
  type        = number
  default     = 2
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the EKS node group"
  type        = number
  default     = 10
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the EKS node group"
  type        = number
  default     = 3
}

# RDS Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "db_allocated_storage" {
  description = "The allocated storage in gigabytes"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "The upper limit to which RDS can automatically scale"
  type        = number
  default     = 1000
}

variable "db_backup_retention_period" {
  description = "The days to retain backups for"
  type        = number
  default     = 30
}

variable "db_maintenance_window" {
  description = "The window to perform maintenance in"
  type        = string
  default     = "sun:03:00-sun:04:00"
}

variable "db_backup_window" {
  description = "The daily time range during which backups are created"
  type        = string
  default     = "04:00-05:00"
}

# ElastiCache Configuration
variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 2
}

variable "redis_parameter_group_name" {
  description = "Name of the parameter group to associate with this cache cluster"
  type        = string
  default     = "default.redis7"
}

variable "redis_port" {
  description = "The port number on which each cache node accepts connections"
  type        = number
  default     = 6379
}

# Application Configuration
variable "domain_name" {
  description = "Primary domain name for the application"
  type        = string
}

variable "alternative_domain_names" {
  description = "Alternative domain names for the application"
  type        = list(string)
  default     = []
}

variable "certificate_arn" {
  description = "ARN of the SSL certificate in ACM"
  type        = string
  default     = ""
}

variable "hosted_zone_id" {
  description = "Route 53 hosted zone ID"
  type        = string
  default     = ""
}

# Monitoring Configuration
variable "prometheus_retention_days" {
  description = "Prometheus data retention period in days"
  type        = number
  default     = 90
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}

variable "alertmanager_slack_webhook" {
  description = "Slack webhook URL for alertmanager notifications"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alertmanager_pagerduty_key" {
  description = "PagerDuty integration key for alertmanager"
  type        = string
  sensitive   = true
  default     = ""
}

# Application Secrets
variable "anthropic_api_key" {
  description = "Anthropic API key for Claude integration"
  type        = string
  sensitive   = true
}

variable "jwt_secret_key" {
  description = "Secret key for JWT token signing"
  type        = string
  sensitive   = true
}

variable "app_secret_key" {
  description = "Application secret key"
  type        = string
  sensitive   = true
}

# Database Secrets
variable "db_master_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "redis_auth_token" {
  description = "Redis authentication token"
  type        = string
  sensitive   = true
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "backup_schedule" {
  description = "Cron schedule for automated backups"
  type        = string
  default     = "0 2 * * *"
}

# Auto Scaling Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

variable "enable_vertical_pod_autoscaler" {
  description = "Enable vertical pod autoscaler"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_pod_security_policy" {
  description = "Enable pod security policies"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable network policies"
  type        = bool
  default     = true
}

variable "enable_secrets_encryption" {
  description = "Enable secrets encryption at rest"
  type        = bool
  default     = true
}

# Cost Management
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_percentage" {
  description = "Percentage of spot instances in node group"
  type        = number
  default     = 30
}

# Logging Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "enable_eks_control_plane_logs" {
  description = "Enable EKS control plane logging"
  type        = bool
  default     = true
}

# Disaster Recovery Configuration
variable "enable_multi_az_deployment" {
  description = "Enable multi-AZ deployment for high availability"
  type        = bool
  default     = true
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

variable "backup_region" {
  description = "Secondary region for backup replication"
  type        = string
  default     = "us-east-1"
}

# Performance Configuration
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval in seconds"
  type        = number
  default     = 60
}

variable "performance_insights_retention" {
  description = "Performance insights retention period in days"
  type        = number
  default     = 7
}

# Tagging Configuration
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Feature Flags
variable "enable_blue_green_deployment" {
  description = "Enable blue-green deployment capability"
  type        = bool
  default     = true
}

variable "enable_canary_deployment" {
  description = "Enable canary deployment capability"
  type        = bool
  default     = true
}

variable "enable_auto_rollback" {
  description = "Enable automatic rollback on deployment failure"
  type        = bool
  default     = true
}
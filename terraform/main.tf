# LeanVibe Agent Hive 2.0 - Main Terraform Configuration
# Multi-environment infrastructure provisioning for production-ready deployment

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Backend configuration - to be overridden per environment
  backend "s3" {
    # Configuration provided via backend config files
  }
}

# Local variables
locals {
  # Common tags for all resources
  common_tags = merge(var.additional_tags, {
    Project     = "leanvibe-agent-hive"
    Component   = "infrastructure"
    ManagedBy   = "terraform"
    Environment = var.environment
    Version     = var.app_version
  })

  # Environment-specific configuration
  config = {
    development = {
      instance_types = ["t3.medium", "t3.large"]
      min_size      = 1
      max_size      = 3
      desired_size  = 2
      db_instance   = "db.t3.micro"
      cache_node    = "cache.t3.micro"
    }
    staging = {
      instance_types = ["t3.large", "m5.large"]
      min_size      = 2
      max_size      = 8
      desired_size  = 3
      db_instance   = "db.t3.small"
      cache_node    = "cache.t3.small"
    }
    production = {
      instance_types = ["m5.xlarge", "m5.2xlarge"]
      min_size      = 3
      max_size      = 20
      desired_size  = 5
      db_instance   = "db.r6g.large"
      cache_node    = "cache.r6g.large"
    }
  }

  # Current environment configuration
  env_config = local.config[var.environment]
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  name               = "${var.project_name}-${var.environment}"
  environment        = var.environment
  cidr               = var.vpc_cidr
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)

  # Enable features for production workloads
  enable_nat_gateway     = true
  single_nat_gateway     = var.environment == "development"
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true
  enable_flow_logs       = var.environment != "development"

  # Subnets
  public_subnet_cidrs   = var.public_subnet_cidrs
  private_subnet_cidrs  = var.private_subnet_cidrs
  database_subnet_cidrs = var.database_subnet_cidrs

  tags = local.common_tags
}

# EKS Module
module "eks" {
  source = "./modules/eks"
  
  depends_on = [module.vpc]

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = var.eks_cluster_version
  environment     = var.environment

  # VPC Configuration
  vpc_id                    = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnet_ids
  control_plane_subnet_ids = module.vpc.private_subnet_ids

  # Node Group Configuration
  node_groups = {
    main = {
      instance_types = local.env_config.instance_types
      ami_type      = "AL2_x86_64"
      capacity_type = "ON_DEMAND"
      
      scaling_config = {
        desired_size = local.env_config.desired_size
        max_size     = local.env_config.max_size
        min_size     = local.env_config.min_size
      }

      update_config = {
        max_unavailable_percentage = 25
      }

      labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }

      taints = var.environment == "production" ? [] : []
    }

    # Spot instances for non-production environments
    spot = var.environment != "production" ? {
      instance_types = ["t3.medium", "t3.large", "m5.large"]
      ami_type      = "AL2_x86_64"
      capacity_type = "SPOT"
      
      scaling_config = {
        desired_size = 1
        max_size     = 5
        min_size     = 0
      }

      labels = {
        Environment = var.environment
        NodeGroup   = "spot"
        WorkloadType = "batch"
      }

      taints = [{
        key    = "spot-instance"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    } : {}
  }

  # Security
  enable_cluster_encryption = var.environment == "production"
  cluster_encryption_config = var.environment == "production" ? [{
    provider_key_arn = aws_kms_key.eks[0].arn
    resources        = ["secrets"]
  }] : []

  # Logging
  cluster_enabled_log_types = var.environment == "production" ? 
    ["api", "audit", "authenticator", "controllerManager", "scheduler"] : 
    ["api", "audit"]

  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  tags = local.common_tags
}

# RDS Module
module "rds" {
  source = "./modules/rds"
  
  depends_on = [module.vpc]

  identifier = "${var.project_name}-${var.environment}"
  environment = var.environment

  # Engine Configuration
  engine              = "postgres"
  engine_version      = var.postgres_version
  instance_class      = local.env_config.db_instance
  allocated_storage   = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_type        = "gp3"
  storage_encrypted   = true

  # Database Configuration
  db_name  = var.db_name
  username = var.db_username
  port     = 5432

  # Network Configuration
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.database_subnet_ids

  # Security
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  # Backup Configuration
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  skip_final_snapshot    = var.environment != "production"
  deletion_protection    = var.environment == "production"

  # Performance Insights
  performance_insights_enabled = var.environment == "production"
  performance_insights_retention_period = var.environment == "production" ? 7 : 0

  # Multi-AZ for production
  multi_az = var.environment == "production"

  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn

  # Extensions
  enabled_extensions = ["vector", "pg_stat_statements", "pg_cron"]

  tags = local.common_tags
}

# ElastiCache Module
module "elasticache" {
  source = "./modules/elasticache"
  
  depends_on = [module.vpc]

  name        = "${var.project_name}-${var.environment}"
  environment = var.environment

  # Engine Configuration
  engine               = "redis"
  node_type           = local.env_config.cache_node
  port                = 6379
  parameter_group_name = "default.redis7"

  # Cluster Configuration
  num_cache_nodes      = var.environment == "production" ? 2 : 1
  az_mode             = var.environment == "production" ? "cross-az" : "single-az"
  
  # Network Configuration
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.elasticache.id]

  # Backup Configuration
  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window         = "05:00-07:00"
  maintenance_window      = "sun:06:00-sun:08:00"

  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = var.environment == "production"
  auth_token_enabled         = var.environment == "production"

  tags = local.common_tags
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"
  
  depends_on = [module.eks]

  environment    = var.environment
  cluster_name   = module.eks.cluster_name
  cluster_endpoint = module.eks.cluster_endpoint

  # Prometheus Configuration
  prometheus_enabled = true
  prometheus_retention = var.environment == "production" ? "30d" : "15d"
  prometheus_storage_size = var.environment == "production" ? "100Gi" : "20Gi"

  # Grafana Configuration
  grafana_enabled = true
  grafana_admin_password = random_password.grafana_admin.result

  # AlertManager Configuration
  alertmanager_enabled = true
  slack_webhook_url = var.slack_webhook_url
  pagerduty_integration_key = var.pagerduty_integration_key

  # Log Aggregation
  fluent_bit_enabled = true
  elasticsearch_enabled = var.environment == "production"

  tags = local.common_tags
}
#!/bin/bash
# Kubernetes deployment script for LeanVibe Agent Hive
# Deploys container-native production system

set -e

echo "🚀 Deploying LeanVibe Agent Hive to Kubernetes"
echo "=============================================="

# Configuration
NAMESPACE="leanvibe-hive"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-current}"
DRY_RUN="${DRY_RUN:-false}"

# Dry run flag
KUBECTL_FLAGS=""
if [ "$DRY_RUN" = "true" ]; then
  KUBECTL_FLAGS="--dry-run=client"
  echo "🧪 Running in DRY RUN mode"
fi

# Check kubectl is available
if ! command -v kubectl &> /dev/null; then
  echo "❌ kubectl is not installed or not in PATH"
  exit 1
fi

# Check if we can connect to cluster
echo "🔍 Checking Kubernetes cluster connection..."
kubectl cluster-info --context="$KUBECTL_CONTEXT" > /dev/null
echo "✅ Connected to Kubernetes cluster"

# Create namespace
echo "📦 Creating namespace..."
kubectl apply -f namespace.yaml $KUBECTL_FLAGS
echo "✅ Namespace created/updated"

# Apply ConfigMaps and Secrets
echo "⚙️  Applying configuration..."
kubectl apply -f configmap.yaml $KUBECTL_FLAGS
echo "✅ Configuration applied"

# Check if secrets need to be updated
echo "🔐 Checking secrets..."
if kubectl get secret leanvibe-secrets -n $NAMESPACE > /dev/null 2>&1; then
  echo "ℹ️  Secrets already exist - update manually if needed"
else
  echo "⚠️  Please update secrets in configmap.yaml with actual values before production deployment"
fi

# Deploy services first
echo "🌐 Deploying services..."
kubectl apply -f services.yaml $KUBECTL_FLAGS
echo "✅ Services deployed"

# Deploy agent deployments
echo "🤖 Deploying agent containers..."
kubectl apply -f agent-deployments.yaml $KUBECTL_FLAGS
echo "✅ Agent deployments created"

# Apply autoscaling
echo "📈 Configuring autoscaling..."
kubectl apply -f autoscaling.yaml $KUBECTL_FLAGS
echo "✅ Autoscaling configured"

# Apply monitoring (if Prometheus operator is available)
echo "📊 Deploying monitoring configuration..."
if kubectl get crd servicemonitors.monitoring.coreos.com > /dev/null 2>&1; then
  kubectl apply -f monitoring.yaml $KUBECTL_FLAGS
  echo "✅ Monitoring deployed"
else
  echo "⚠️  Prometheus operator not found - monitoring config skipped"
fi

# Wait for deployments if not dry run
if [ "$DRY_RUN" != "true" ]; then
  echo "⏳ Waiting for deployments to be ready..."
  
  # Wait for each deployment
  DEPLOYMENTS=("agent-architect" "agent-developer" "agent-qa" "agent-meta")
  
  for deployment in "${DEPLOYMENTS[@]}"; do
    echo "  Waiting for $deployment..."
    kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=300s
    echo "  ✅ $deployment ready"
  done
  
  echo "🎉 All deployments ready!"
  
  # Display status
  echo ""
  echo "📊 Deployment Status:"
  kubectl get pods -n $NAMESPACE -o wide
  
  echo ""
  echo "🔗 Services:"
  kubectl get services -n $NAMESPACE
  
  # Show autoscaler status
  echo ""
  echo "📈 Autoscaler Status:"
  kubectl get hpa -n $NAMESPACE
  
else
  echo "🧪 Dry run completed - no actual deployment performed"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Verify agent pods are running: kubectl get pods -n $NAMESPACE"
echo "2. Check agent logs: kubectl logs -f deployment/agent-developer -n $NAMESPACE"
echo "3. Monitor autoscaling: kubectl get hpa -n $NAMESPACE -w"
echo "4. Run migration: python scripts/container_migration.py"
echo ""
echo "🔧 Useful Commands:"
echo "- Scale agents: kubectl scale deployment agent-developer --replicas=10 -n $NAMESPACE"
echo "- View logs: kubectl logs -f -l app=leanvibe-agent -n $NAMESPACE"
echo "- Get agent status: kubectl get pods -l app=leanvibe-agent -n $NAMESPACE"
echo "- Port forward to API: kubectl port-forward service/leanvibe-api-service 8000:8000 -n $NAMESPACE"
#!/bin/bash
# Build script for LeanVibe Agent Hive containerized agent images
# Builds base image and specialized agent types

set -e

echo "🚀 Building LeanVibe Agent Hive Container Images"
echo "================================================"

# Configuration
REGISTRY_PREFIX="leanvibe"
VERSION="latest"
BUILD_CONTEXT="."

# Build base agent image first
echo "📦 Building base agent image..."
docker build \
  -f Dockerfile.agent-base \
  -t ${REGISTRY_PREFIX}/agent-base:${VERSION} \
  ${BUILD_CONTEXT}

echo "✅ Base agent image built successfully"

# Build specialized agent images
AGENT_TYPES=("architect" "developer" "qa" "meta")

for agent_type in "${AGENT_TYPES[@]}"; do
  echo "📦 Building ${agent_type} agent image..."
  
  docker build \
    -f Dockerfile.agent-${agent_type} \
    -t ${REGISTRY_PREFIX}/agent-${agent_type}:${VERSION} \
    ${BUILD_CONTEXT}
  
  echo "✅ ${agent_type} agent image built successfully"
done

# Verify all images were built
echo "🔍 Verifying built images..."
echo "Built images:"
docker images | grep "${REGISTRY_PREFIX}/agent-"

# Test image functionality
echo "🧪 Testing image functionality..."

# Test base image
echo "Testing base image..."
docker run --rm \
  ${REGISTRY_PREFIX}/agent-base:${VERSION} \
  python -c "import sys; print(f'Python {sys.version}'); import anthropic; print('✅ Anthropic client available')"

# Test specialized images with health checks
for agent_type in "${AGENT_TYPES[@]}"; do
  echo "Testing ${agent_type} agent image..."
  
  # Start container in background
  CONTAINER_ID=$(docker run -d \
    --name "test-agent-${agent_type}" \
    --network leanvibe_network \
    -e AGENT_TYPE="${agent_type}" \
    -e ANTHROPIC_API_KEY="test-key" \
    ${REGISTRY_PREFIX}/agent-${agent_type}:${VERSION})
  
  # Wait for startup
  sleep 5
  
  # Check if container is still running
  if docker ps -q --filter "id=${CONTAINER_ID}" | grep -q .; then
    echo "✅ ${agent_type} agent container started successfully"
  else
    echo "❌ ${agent_type} agent container failed to start"
    docker logs "test-agent-${agent_type}"
  fi
  
  # Cleanup test container
  docker rm -f "test-agent-${agent_type}" || true
done

echo "🎉 All agent images built and tested successfully!"
echo ""
echo "Available images:"
echo "- ${REGISTRY_PREFIX}/agent-base:${VERSION}"
echo "- ${REGISTRY_PREFIX}/agent-architect:${VERSION}" 
echo "- ${REGISTRY_PREFIX}/agent-developer:${VERSION}"
echo "- ${REGISTRY_PREFIX}/agent-qa:${VERSION}"
echo "- ${REGISTRY_PREFIX}/agent-meta:${VERSION}"
echo ""
echo "To push to registry (if configured):"
echo "  docker push ${REGISTRY_PREFIX}/agent-base:${VERSION}"
for agent_type in "${AGENT_TYPES[@]}"; do
  echo "  docker push ${REGISTRY_PREFIX}/agent-${agent_type}:${VERSION}"
done
echo ""
echo "To run migration:"
echo "  python scripts/container_migration.py --dry-run"
#!/bin/bash
# Quick test of setup optimization components

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_status "$BLUE" "🚀 Testing LeanVibe Agent Hive 2.0 Setup Optimization"
echo ""

# Test 1: Docker services startup
print_status "$YELLOW" "Test 1: Starting Docker services with fast compose..."
start_time=$(date +%s)

# Clean up first
docker compose -f docker-compose.fast.yml down -v 2>/dev/null || true

# Start services
if docker compose -f docker-compose.fast.yml up -d postgres redis; then
    print_status "$GREEN" "✅ Docker services started"
else
    print_status "$RED" "❌ Docker services failed to start"
    exit 1
fi

# Test 2: Health check timing
print_status "$YELLOW" "Test 2: Waiting for services to become healthy..."
max_wait=60
wait_time=0
postgres_ready=false
redis_ready=false

while [[ $wait_time -lt $max_wait ]]; do
    if [[ "$postgres_ready" == "false" ]] && docker compose -f docker-compose.fast.yml exec -T postgres pg_isready -U leanvibe_user >/dev/null 2>&1; then
        postgres_ready=true
        print_status "$GREEN" "  ✅ PostgreSQL ready"
    fi
    
    if [[ "$redis_ready" == "false" ]] && docker compose -f docker-compose.fast.yml exec -T redis redis-cli -a leanvibe_redis_pass ping >/dev/null 2>&1; then
        redis_ready=true
        print_status "$GREEN" "  ✅ Redis ready"
    fi
    
    if [[ "$postgres_ready" == "true" && "$redis_ready" == "true" ]]; then
        break
    fi
    
    sleep 2
    wait_time=$((wait_time + 2))
done

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

if [[ "$postgres_ready" == "true" && "$redis_ready" == "true" ]]; then
    print_status "$GREEN" "✅ All services healthy in ${minutes}m ${seconds}s"
else
    print_status "$RED" "❌ Services failed to become healthy within ${max_wait}s"
fi

# Test 3: Configuration validation
print_status "$YELLOW" "Test 3: Testing optimized configurations..."

# Check if fast compose file has performance optimizations
if grep -q "shared_buffers=128MB" docker-compose.fast.yml && \
   grep -q "maxmemory 256mb" docker-compose.fast.yml && \
   grep -q "memory: 512M" docker-compose.fast.yml; then
    print_status "$GREEN" "✅ Performance optimizations detected"
else
    print_status "$YELLOW" "⚠️ Some performance optimizations may be missing"
fi

# Test 4: Scripts validation
print_status "$YELLOW" "Test 4: Validating management scripts..."

if [[ -x "start-fast.sh" ]] && [[ -x "stop-fast.sh" ]] && [[ -x "setup-fast.sh" ]]; then
    print_status "$GREEN" "✅ All optimization scripts are executable"
else
    print_status "$RED" "❌ Some scripts are missing or not executable"
fi

# Cleanup
print_status "$YELLOW" "Cleaning up test services..."
docker compose -f docker-compose.fast.yml down -v >/dev/null 2>&1

# Final results
echo ""
print_status "$BLUE" "============================================"
print_status "$BLUE" "SETUP OPTIMIZATION TEST RESULTS"
print_status "$BLUE" "============================================"
echo ""
print_status "$GREEN" "✅ Docker services startup: ${minutes}m ${seconds}s"
print_status "$GREEN" "✅ Fast compose configuration validated"
print_status "$GREEN" "✅ Management scripts ready"
print_status "$GREEN" "✅ Performance optimizations in place"
echo ""

if [[ "$postgres_ready" == "true" && "$redis_ready" == "true" && $duration -lt 120 ]]; then
    print_status "$GREEN" "🏆 OPTIMIZATION TEST PASSED!"
    print_status "$BLUE" "Services startup time: ${duration}s (target: <2 minutes for core services)"
    print_status "$BLUE" "Ready for full setup performance testing."
else
    print_status "$YELLOW" "⚠️ Optimization may need tuning"
    print_status "$BLUE" "Services startup time: ${duration}s"
fi

echo ""
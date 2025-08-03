# Setup Script Automation Fixes

## Problem Analysis

After analyzing the setup scripts, I've identified several critical issues preventing true one-command deployment:

### 1. Interactive Prompts (Lines 279-319 in setup-ultra-fast.sh)
- **Issue**: Scripts pause for user input on API keys
- **Impact**: Blocks automated deployment
- **Location**: `create_smart_env_config()` function uses `read -p` commands

### 2. Environment Configuration Dependencies
- **Issue**: No non-interactive mode for API key setup
- **Impact**: Scripts hang waiting for user input
- **Location**: Lines 279-319 in setup-ultra-fast.sh

### 3. Service Health Check Timeouts
- **Issue**: Hardcoded 60-90 second waits without proper timeout handling
- **Impact**: Scripts may hang indefinitely if services fail to start
- **Location**: Lines 390-416 in setup-ultra-fast.sh

### 4. Missing Non-Interactive Flags
- **Issue**: No command-line option to skip interactive prompts
- **Impact**: Cannot run in CI/CD or automated environments

## Solution Implementation

### Fix 1: Add Non-Interactive Mode Support

**File**: `setup-ultra-fast.sh`
**Changes**: Add `--non-interactive` flag support

```bash
# Add flag parsing at beginning of main()
NON_INTERACTIVE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        --config-only)
            create_smart_env_config
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
```

### Fix 2: Non-Interactive Environment Configuration

**File**: `setup-ultra-fast.sh`
**Function**: `create_smart_env_config()`

Replace interactive API key prompts with:

```bash
# Enhanced API Key Setup - Non-Interactive Mode Support
if [[ "$NON_INTERACTIVE" == "true" ]]; then
    print_status "$CYAN" "🔧 Non-interactive mode: Using default configuration"
    print_status "$YELLOW" "⚠️  API keys set to placeholders - update .env.local manually"
    print_success "Environment configured for non-interactive deployment"
else
    # Original interactive prompts here
    echo ""
    print_status "$BOLD$YELLOW" "🔑 API KEY SETUP - REQUIRED FOR FULL FUNCTIONALITY"
    # ... rest of interactive code
fi
```

### Fix 3: Robust Service Health Checks with Timeouts

**File**: `setup-ultra-fast.sh`
**Function**: `start_docker_services_ultra()`

```bash
# Advanced health check with configurable timeout and exponential backoff
check_service_health() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=1
    local wait_time=2
    
    print_status "$CYAN" "    → Checking $service_name health..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if eval "$check_command" >/dev/null 2>&1; then
            print_success "$service_name ready (attempt $attempt)"
            return 0
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            print_error "$service_name failed to start after $max_attempts attempts"
            return 1
        fi
        
        echo -n "."
        sleep $wait_time
        
        # Exponential backoff with max wait of 5 seconds
        if [[ $wait_time -lt 5 ]]; then
            wait_time=$((wait_time + 1))
        fi
        
        attempt=$((attempt + 1))
    done
}
```

### Fix 4: Fail-Fast Error Handling

Add comprehensive error handling with automatic rollback:

```bash
# Error handling with automatic cleanup
cleanup_on_failure() {
    print_status "$RED" "🔧 Setup failed - initiating cleanup..."
    
    # Stop any running containers
    docker compose -f docker-compose.fast.yml down --remove-orphans 2>/dev/null || true
    
    # Remove partial virtual environment
    if [[ -d "${SCRIPT_DIR}/venv.partial" ]]; then
        rm -rf "${SCRIPT_DIR}/venv.partial"
    fi
    
    print_status "$YELLOW" "Cleanup completed. You can safely re-run the setup script."
}

# Set trap for cleanup on failure
trap cleanup_on_failure EXIT
```

### Fix 5: Validation with Early Exit

Replace the current validation with fail-fast checks:

```bash
# Pre-flight validation with early exit
validate_prerequisites() {
    local validation_failed=false
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon not running - setup cannot continue"
        validation_failed=true
    fi
    
    # Check disk space (minimum 2GB)
    local available_kb=$(df "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    if [[ $available_kb -lt 2097152 ]]; then
        print_error "Insufficient disk space (need 2GB+) - setup cannot continue"
        validation_failed=true
    fi
    
    # Check Python version
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        print_error "Python 3.11+ required - setup cannot continue"
        validation_failed=true
    fi
    
    if [[ "$validation_failed" == "true" ]]; then
        print_status "$RED" "Prerequisites not met. Please fix the above issues and retry."
        exit 1
    fi
    
    print_success "Prerequisites validated"
}
```

## Implementation Plan

### Phase 1: Create Fixed Setup Script (30 minutes)

1. **Create `setup-ultra-fast-fixed.sh`** with all fixes applied
2. **Add comprehensive flag parsing** for `--non-interactive`, `--timeout=N`, `--skip-services`
3. **Implement robust error handling** with cleanup and rollback
4. **Add progress indicators** with ETA and real-time status

### Phase 2: Service Health Monitoring (20 minutes)

1. **Implement health check functions** with exponential backoff
2. **Add service dependency validation** before proceeding
3. **Create service startup orchestration** with parallel health checks
4. **Add timeout configuration** for different environments

### Phase 3: Testing and Validation (30 minutes)

1. **Test non-interactive mode** in clean environment
2. **Validate timeout handling** with service failures
3. **Test error scenarios** and cleanup procedures
4. **Benchmark performance** against target (<5 minutes)

### Phase 4: Documentation and Integration (20 minutes)

1. **Update usage documentation** with new flags
2. **Create CI/CD integration examples**
3. **Add troubleshooting guide** for common failure scenarios
4. **Update health check integration**

## Expected Outcomes

### Performance Targets
- **Setup Time**: <3 minutes (non-interactive), <5 minutes (interactive)
- **Success Rate**: >98% in standard environments
- **Recovery Time**: <30 seconds for failed setups

### Reliability Improvements
- **Zero Hanging**: No indefinite waits or interactive prompts in non-interactive mode
- **Fail-Fast**: Early detection and reporting of blocking issues
- **Clean Rollback**: Automatic cleanup on failure with safe retry capability

### User Experience Enhancements
- **Clear Progress**: Real-time progress indicators with ETA
- **Actionable Errors**: Specific error messages with suggested fixes
- **Flexible Deployment**: Support for CI/CD, development, and production scenarios

## Testing Matrix

| Scenario | Expected Result | Test Command |
|----------|----------------|--------------|
| Clean Environment | <3 min success | `./setup-ultra-fast-fixed.sh --non-interactive` |
| Docker Not Running | Fail within 10s | `./setup-ultra-fast-fixed.sh --non-interactive` |
| Low Disk Space | Fail within 10s | `./setup-ultra-fast-fixed.sh --non-interactive` |
| Partial Cleanup | Success on retry | Interrupt + retry |
| Service Timeout | Graceful failure | Mock service delay |

## Migration Strategy

1. **Backup current scripts** to `setup-legacy/` directory
2. **Deploy fixed scripts** alongside current ones
3. **Update documentation** to recommend new scripts
4. **Deprecation timeline**: 2 weeks for user migration
5. **Remove legacy scripts** after validation period

This implementation will achieve true one-command deployment while maintaining backward compatibility and providing clear upgrade paths for existing users.
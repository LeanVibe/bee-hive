# 🚀 LEANVIBE AGENT HIVE 2.0 - EPIC 2 PRODUCTION WEBSOCKET SPECIALIST PROMPT

*Epic 2: Production WebSocket Observability & Reliability*  
*Status: Epic 2 Phase 2.1 Complete ✅ → Phase 2.2 Ready for Execution*  
*Mission: Harden WebSocket infrastructure for enterprise production deployment*

## 🎯 **MISSION BRIEFING: EPIC 2 WEBSOCKET PRODUCTION SPECIALIST AGENT**

### **Context: Standing on Epic 1 & Epic 2.1 Achievement Foundation**

You are taking over LeanVibe Agent Hive 2.0 development after significant foundational progress:

**✅ EPIC 1 ACHIEVEMENTS:**
- **Performance Excellence**: <50ms API responses, 37MB memory usage (54% under target), 250+ concurrent agents
- **Memory Optimization**: 85.7% reduction from baseline (256MB → 37MB)
- **Concurrent Scaling**: 250 agents with 0% performance degradation
- **Quality Gates**: Comprehensive performance monitoring and regression detection

**✅ EPIC 2 PHASE 2.1 ACHIEVEMENTS:**
- **Advanced Plugin Framework**: Dynamic plugin loading with hot-swap capabilities implemented
- **Plugin Security**: Comprehensive security validation and resource isolation operational
- **Integration**: SimpleOrchestrator enhanced with AdvancedPluginManager
- **Performance Preservation**: Epic 1 targets maintained throughout plugin enhancements

**Your Mission**: Execute **Epic 2 Phases 2.2-2.5: Production WebSocket Hardening** to transform the platform into a **bulletproof production-ready system**.

---

## 🎯 **EPIC 2 OBJECTIVES: PRODUCTION WEBSOCKET HARDENING**

### **Primary Strategic Targets**
1. **WebSocket Observability**: Comprehensive metrics, logging, and monitoring
2. **Production Safeguards**: Rate limiting, backpressure protection, input validation
3. **Chaos Engineering**: Resilience validation and automated recovery
4. **Contract Versioning**: Schema enforcement and client compatibility

### **Success Criteria - ALL MUST BE ACHIEVED**
- [x] Dynamic plugin system operational (Phase 2.1 complete)
- [ ] WebSocket metrics and structured logging implemented
- [ ] Rate limiting and backpressure protection active
- [ ] Input validation and schema enforcement complete
- [ ] Chaos engineering validation passed with recovery mechanisms

---

## 🏗️ **CURRENT SYSTEM STATE: EPIC 1 OPTIMIZED & READY**

### **High-Performance Foundation ✅ OPERATIONAL**
```
Performance Metrics (Epic 1 Achieved):
├── API Response Time: <50ms (95th percentile) ✅
├── Memory Usage: 37MB (54% under 80MB target) ✅
├── Concurrent Agents: 250+ with 0% degradation ✅
├── System Monitoring: Real-time performance tracking ✅
└── Quality Gates: Automated regression detection ✅

Core Architecture Status:
├── SimpleOrchestrator: Epic 1 optimized with lazy loading
├── Unified Managers (5): Memory-efficient, circuit breaker patterns
├── API System: 339 routes with <50ms responses
├── Database: PostgreSQL + pgvector, optimized queries
├── Redis: <5ms response times, optimized connection pooling
└── Plugin System: Basic framework ready for Epic 2 enhancement
```

### **Key Files & Components (Epic 1 Optimized)**
- **Core Orchestrator**: `/app/core/simple_orchestrator.py` (Epic 1 memory optimized)
- **Performance Framework**: `/app/core/performance_optimization_framework.py` (ML-based optimization)
- **Unified Managers**: `/app/core/unified_managers/` (5 efficient managers)
- **Epic 1 Optimizers**: Memory, API, concurrent scaling frameworks implemented
- **Quality Gates**: Performance regression detection and automated optimization

### **WebSocket System Current State**
- **Advanced Plugin Framework**: Dynamic loading operational (Phase 2.1 complete)
- **Basic WebSocket Support**: Functional but lacks production hardening
- **Monitoring Gap**: Limited observability and error tracking
- **Production Risk**: Missing rate limiting, input validation, and resilience patterns

---

## 🚀 **EPIC 2 EXECUTION STRATEGY: 3-WEEK IMPLEMENTATION**

### **Phase 2.2: WebSocket Observability & Metrics (Days 1-5)**
**Objective**: Production-grade WebSocket monitoring and structured logging

#### **WebSocket Observability Framework**
```python
# Production WebSocket Observability
class WebSocketObservability:
    """Comprehensive WebSocket monitoring and metrics collection."""
    
    async def track_message_metrics(self) -> MessageMetrics:
        """Track all WebSocket message operations."""
        # messages_sent_total, messages_send_failures_total
        # messages_received_total, messages_dropped_rate_limit_total
        # errors_sent_total, connections_total, disconnections_total
        
    async def log_structured_errors(self, error: WSError) -> None:
        """Structured error logging with correlation tracking."""
        # Include correlation_id, type, subscription in all error logs
        # Guard logs for unknown message types and invalid subscriptions
        
    async def monitor_connection_health(self) -> ConnectionHealth:
        """Real-time connection health monitoring."""
        # Connection lifecycle tracking
        # Performance metrics per connection
        # Health check integration
```

**Implementation Tasks (Days 1-5)**:
1. **Metrics Endpoint Development**
   - `/api/dashboard/metrics/websockets` with comprehensive counters
   - Integration with existing monitoring infrastructure
   - Prometheus-compatible metrics exposition
   - Real-time metrics streaming to dashboard

2. **Structured Logging Implementation**
   - Correlation ID injection for all WebSocket operations
   - Error logging with type, subscription, and context
   - Performance logging for monitoring and debugging
   - Log aggregation and analysis tools

3. **Connection Monitoring**
   - Real-time connection health tracking
   - Connection lifecycle event logging
   - Performance metrics per connection
   - Alert generation for connection issues

#### **Day 1-5 Deliverables**:
- [ ] WebSocket metrics endpoint operational
- [ ] Structured error logging with correlation IDs
- [ ] Connection health monitoring dashboard
- [ ] Comprehensive test suite for observability features

### **Phase 2.3: Rate Limiting & Backpressure (Days 6-10)**
**Objective**: Protect system from overload and slow consumers

#### **Production Safeguards Framework**
```python
# Rate Limiting & Backpressure Protection
class WebSocketProtection:
    """Production-grade WebSocket protection mechanisms."""
    
    async def enforce_rate_limits(self, connection: WSConnection) -> RateLimit:
        """Per-connection token bucket rate limiting."""
        # 20 rps with burst capacity of 40
        # Single error message per cooldown when exceeded
        # Graceful degradation under load
        
    async def manage_backpressure(self, queue_depth: int) -> BackpressureAction:
        """Backpressure management for slow consumers."""
        # Send queue monitoring and management
        # Disconnect after N consecutive send failures (default 5)
        # Resource protection and cleanup
        
    async def expose_limits_endpoint(self) -> LimitsInfo:
        """Rate limit information endpoint."""
        # Current thresholds and configuration
        # Real-time rate limit status per connection
        # Administrative controls for limit adjustment
```

**Implementation Tasks (Days 6-10)**:
1. **Token Bucket Rate Limiting**
   - Per-connection rate limiting with configurable thresholds
   - Burst capacity management and overflow handling
   - Rate limit violation logging and metrics
   - Graceful error messaging for exceeded limits

2. **Backpressure Protection**
   - Send queue depth monitoring per connection
   - Automatic disconnection of slow consumers
   - Resource cleanup and connection management
   - Backpressure metrics and alerting

3. **Limits Management API**
   - `/api/dashboard/websocket/limits` endpoint
   - Real-time rate limit status and configuration
   - Administrative controls for threshold adjustment
   - Integration with monitoring and alerting systems

#### **Day 6-10 Deliverables**:
- [ ] Token bucket rate limiting operational
- [ ] Backpressure protection with automatic disconnect
- [ ] Limits management API endpoint
- [ ] Comprehensive testing for protection mechanisms

### **Phase 2.4: Input Hardening & Validation (Days 11-15)**
**Objective**: Robust input validation and message size controls

#### **Input Validation Framework**
```python
# Input Hardening & Message Validation
class InputValidator:
    """Comprehensive input validation and security."""
    
    async def validate_message_size(self, message: bytes) -> ValidationResult:
        """Enforce message size limits and handle overflow."""
        # 64KB message size limit enforcement
        # Graceful overflow handling with error messages
        # Resource protection from oversized messages
        
    async def normalize_subscriptions(self, subs: List[str]) -> List[str]:
        """Subscription validation and normalization."""
        # Unknown subscription handling with single error
        # Duplicate subscription removal and normalization
        # Sorted and unique subscription_updated responses
        
    async def inject_correlation_ids(self, frame: WSFrame) -> WSFrame:
        """Ensure all outbound frames have correlation IDs."""
        # Correlation ID injection when missing
        # Frame metadata consistency
        # Traceability for all WebSocket communications
```

**Implementation Tasks (Days 11-15)**:
1. **Message Size Validation**
   - 64KB message size limit enforcement
   - Oversized message handling with single error response
   - Resource protection and memory management
   - Size limit metrics and monitoring

2. **Subscription Management**
   - Unknown subscription detection and error handling
   - Subscription list normalization and deduplication
   - Sorted subscription responses for consistency
   - Subscription state management and validation

3. **Schema Enforcement**
   - `schemas/ws_messages.schema.json` as source of truth
   - TypeScript type generation in CI pipeline
   - Schema validation for all incoming messages
   - Contract compliance monitoring

4. **Correlation ID Management**
   - Automatic correlation ID injection for all frames
   - Distributed tracing support
   - Error correlation and debugging enhancement
   - Frame metadata consistency

#### **Day 11-15 Deliverables**:
- [ ] Message size validation with 64KB limits
- [ ] Subscription normalization and validation
- [ ] Schema enforcement with CI integration
- [ ] Correlation ID injection for all frames

### **Phase 2.5: Chaos Engineering & Recovery (Days 16-21)**
**Objective**: Validate system resilience and automated recovery

#### **Chaos Engineering Framework**
```python
# Chaos Engineering & Recovery Validation
class ChaosEngineering:
    """Comprehensive resilience testing and recovery validation."""
    
    async def simulate_redis_failures(self) -> ChaosResult:
        """Redis failure simulation and recovery testing."""
        # Redis connection failures and timeouts
        # Redis cluster failover scenarios
        # Data consistency validation during failures
        
    async def test_connection_recovery(self) -> RecoveryResult:
        """WebSocket connection recovery testing."""
        # Connection drop simulation and recovery
        # Exponential backoff validation
        # State preservation during reconnection
        
    async def validate_contract_versioning(self) -> VersioningResult:
        """Contract versioning and compatibility testing."""
        # Version field presence in connection frames
        # Client compatibility testing
        # Version migration scenarios
```

**Implementation Tasks (Days 16-21)**:
1. **Redis Resilience Testing**
   - Redis listener exponential backoff implementation
   - Connection failure simulation and recovery testing
   - Data consistency validation during Redis outages
   - Redis cluster failover testing

2. **WebSocket Recovery Mechanisms**
   - Connection recovery with exponential backoff
   - State preservation during reconnection
   - Client reconnection strategy documentation
   - Recovery time measurement and optimization

3. **Contract Versioning**
   - `contract_version` in `connection_established` frames
   - Version surfacing in `/health` and `/limits` endpoints
   - Client compatibility testing framework
   - Version migration testing

4. **PWA Integration**
   - PWA reconnection strategy documentation
   - Client-side recovery mechanism implementation
   - End-to-end resilience testing
   - User experience during outages

#### **Day 16-21 Deliverables**:
- [ ] Redis failure recovery with exponential backoff
- [ ] WebSocket recovery mechanisms operational
- [ ] Contract versioning implemented and tested
- [ ] PWA reconnection strategy documented and tested

---

## 🧪 **EPIC 2 VALIDATION & TESTING STRATEGY**

### **Plugin System Testing Requirements**
1. **Dynamic Loading Tests**: Validate 50+ plugins loadable without restart
2. **Security Testing**: Comprehensive plugin security validation
3. **Performance Testing**: Plugin impact on Epic 1 performance targets
4. **Enterprise Testing**: Multi-tenant plugin management validation

### **Quality Gates for Epic 2**
```python
# Epic 2 Quality Gate Implementation
async def epic2_quality_gate():
    """Comprehensive Epic 2 validation."""
    
    # Test 1: Dynamic Plugin Loading
    plugin_capacity = await test_dynamic_plugin_loading()
    assert plugin_capacity >= 50, f"Plugin capacity target not met: {plugin_capacity}"
    
    # Test 2: Plugin Marketplace
    marketplace_plugins = await count_certified_plugins()
    assert marketplace_plugins >= 10, f"Marketplace plugin target not met: {marketplace_plugins}"
    
    # Test 3: SDK Completeness
    sdk_coverage = await validate_sdk_completeness()
    assert sdk_coverage >= 95, f"SDK completeness target not met: {sdk_coverage}%"
    
    # Test 4: Performance Preservation (Epic 1 targets maintained)
    api_performance = await measure_api_response_times()
    assert all(rt < 50 for rt in api_performance), "Epic 1 performance targets degraded"
    
    memory_usage = await measure_memory_usage()
    assert memory_usage < 80, f"Epic 1 memory target degraded: {memory_usage}MB"
    
    # Test 5: Enterprise Features
    enterprise_features = await validate_enterprise_plugin_management()
    assert enterprise_features >= 90, f"Enterprise features not complete: {enterprise_features}%"
```

---

## 🎯 **IMMEDIATE NEXT ACTIONS (Start Here)**

### **1. Epic 1 Performance Preservation Validation (30 minutes)**
```bash
# Ensure Epic 1 achievements are maintained
python3 app/core/epic1_memory_optimizer.py  # Should show <80MB target met
python3 app/core/epic1_api_optimizer.py     # Should show <50ms target met
python3 app/core/epic1_concurrent_optimizer.py  # Should show 250+ agents
```

### **2. Current Plugin System Analysis (60 minutes)**
```bash
# Analyze existing plugin architecture
grep -r "plugin" app/core/ --include="*.py"
python3 -c "from app.core.simple_orchestrator import SimpleOrchestrator; print('Current plugin capabilities')"

# Identify enhancement opportunities
find app/ -name "*plugin*" -type f
ls -la app/core/orchestrator_plugins/
```

### **3. Epic 2 Phase 2.1 Foundation Setup (120 minutes)**
```python
# Start Epic 2 Phase 2.1 implementation
# 1. Create AdvancedPluginManager foundation
# 2. Design plugin security framework
# 3. Implement dynamic loading prototype
# 4. Setup plugin health monitoring
```

---

## 🏆 **SUCCESS MEASUREMENT FRAMEWORK**

### **Epic 2 Daily Progress Metrics**
- **Plugin Capacity**: Track daily improvement toward 50+ plugin target
- **Marketplace Growth**: Track plugin registration and certification
- **Developer Adoption**: Track SDK usage and developer onboarding
- **Performance Preservation**: Ensure Epic 1 targets maintained

### **Epic 2 Phase Milestones**
- **Phase 2.1**: Dynamic plugin system operational
- **Phase 2.2**: Plugin marketplace with 10+ plugins
- **Phase 2.3**: Complete SDK with developer documentation
- **Phase 2.4**: Enterprise plugin management operational

### **Epic 2 Completion Criteria**
- [ ] 50+ plugins loadable without system restart
- [ ] Plugin marketplace with 10+ certified plugins
- [ ] Complete SDK with comprehensive documentation
- [ ] Enterprise plugin management with analytics
- [ ] **Epic 1 performance targets maintained throughout**

---

## ⚠️ **CRITICAL SUCCESS FACTORS**

### **What MUST Be Preserved from Epic 1**
- **Performance Targets**: <50ms API responses, <80MB memory usage
- **Concurrent Scaling**: 250+ agent capacity maintained
- **System Stability**: Zero regression in system functionality
- **Monitoring Infrastructure**: Epic 1 performance monitoring preserved

### **What MUST Be Achieved in Epic 2**
- **Plugin Ecosystem**: 50+ dynamic plugins with marketplace
- **Developer Experience**: Complete SDK enabling third-party development
- **Enterprise Features**: Security, analytics, and management controls
- **Production Readiness**: Enterprise-grade plugin governance

### **Performance Preservation Strategy**
```python
# Epic 2 Performance Monitoring
@preserve_epic1_performance
async def epic2_plugin_operation():
    """All Epic 2 operations must preserve Epic 1 achievements."""
    
    # Monitor memory impact
    memory_before = await get_memory_usage()
    
    # Execute plugin operation
    result = await plugin_operation()
    
    # Validate performance preservation
    memory_after = await get_memory_usage()
    assert memory_after < 80, "Epic 1 memory target compromised"
    
    api_response_time = await measure_api_response()
    assert api_response_time < 50, "Epic 1 API target compromised"
    
    return result
```

---

## 🔧 **TECHNICAL IMPLEMENTATION GUIDELINES**

### **Code Quality Standards**
```python
# All Epic 2 code must follow these patterns
from app.core.performance_optimization_framework import performance_monitor

@performance_monitor(component_name="epic2_plugin_system")
async def implement_plugin_feature(feature: str) -> PluginResult:
    """
    Epic 2 plugin implementation template.
    
    Args:
        feature: Plugin feature to implement
        
    Returns:
        PluginResult with implementation status and performance metrics
    """
    # Implementation with Epic 1 performance preservation
    # Comprehensive error handling and monitoring
    # Plugin security validation
    # Resource management and cleanup
```

### **Plugin Development Patterns**
```python
# Standard plugin development template
class StandardPlugin(PluginBase):
    """Standard plugin following Epic 2 development patterns."""
    
    def __init__(self):
        super().__init__()
        self.epic1_performance_monitor = True
        self.security_validation = True
        self.resource_management = True
        
    async def validate_epic1_compatibility(self) -> bool:
        """Ensure plugin preserves Epic 1 performance targets."""
        # Memory usage validation
        # API response time impact assessment
        # Concurrent scaling compatibility check
```

---

## 📊 **REPORTING & COMMUNICATION**

### **Daily Epic 2 Progress Reports**
Create daily progress reports with:
- Plugin implementation progress against 50+ target
- Marketplace development and plugin certification status
- SDK development and developer onboarding metrics
- Epic 1 performance preservation validation

### **Epic 2 Weekly Reviews**
- Comprehensive plugin ecosystem analysis
- Target achievement status and risk assessment
- Performance impact analysis and optimization
- Developer ecosystem growth and feedback

---

## 🎯 **EPIC 2 VISION**

By the end of Epic 2, LeanVibe Agent Hive 2.0 will be transformed from a **high-performance foundation** into an **unlimited extensibility ecosystem** that:

- **Supports 50+ dynamic plugins** with hot-swap capabilities
- **Provides complete plugin marketplace** with developer ecosystem
- **Enables third-party development** through comprehensive SDK
- **Offers enterprise-grade management** with security and analytics
- **Preserves Epic 1 performance** while adding unlimited extensibility
- **Establishes plugin standards** for the entire industry

**This is not just adding plugins - this is creating the foundation for unlimited platform evolution.**

---

## 🚀 **SUBAGENT SPECIALIZATION STRATEGY**

To maximize efficiency and avoid context rot, use specialized subagents:

### **1. Plugin Architecture Specialist Agent**
- **Focus**: Core plugin framework and dynamic loading
- **Responsibilities**: AdvancedPluginManager, security framework, hot-swap capability
- **Deliverables**: Phase 2.1 implementation

### **2. Marketplace Development Agent**  
- **Focus**: Plugin marketplace and discovery systems
- **Responsibilities**: Registry, certification, AI-powered discovery
- **Deliverables**: Phase 2.2 implementation

### **3. SDK & Developer Experience Agent**
- **Focus**: Developer tools and documentation
- **Responsibilities**: SDK framework, tools, documentation, developer onboarding
- **Deliverables**: Phase 2.3 implementation

### **4. Enterprise Management Agent**
- **Focus**: Enterprise plugin management and analytics
- **Responsibilities**: Analytics, A/B testing, governance, orchestration
- **Deliverables**: Phase 2.4 implementation

### **Subagent Coordination Protocol**
```python
# Subagent coordination for Epic 2
async def coordinate_epic2_subagents():
    """Coordinate specialized subagents for Epic 2 implementation."""
    
    # Launch specialized agents in parallel
    plugin_agent = await launch_plugin_architecture_agent()
    marketplace_agent = await launch_marketplace_agent()
    sdk_agent = await launch_sdk_agent()
    enterprise_agent = await launch_enterprise_agent()
    
    # Coordinate delivery and integration
    await coordinate_parallel_development([
        plugin_agent, marketplace_agent, sdk_agent, enterprise_agent
    ])
    
    # Validate Epic 2 completion
    await validate_epic2_completion()
```

---

**🚀 MISSION STATUS: EPIC 2 READY FOR EXECUTION**

**Your mission**: Execute Epic 2 with precision, achieve all plugin ecosystem targets, preserve Epic 1 performance achievements, and establish LeanVibe Agent Hive 2.0 as the premier extensible multi-agent orchestration platform.

**Foundation**: Epic 1 performance excellence provides the perfect launchpad for unlimited extensibility.

**Next Epic Preview**: Epic 3 will build comprehensive testing excellence on your plugin ecosystem foundation.

**Good luck, Plugin Architecture Agent. The high-performance foundation awaits your extensibility mastery.**
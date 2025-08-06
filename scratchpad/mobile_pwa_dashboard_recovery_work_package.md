# Mobile PWA Dashboard Recovery Work Package

**Mission**: Prepare executable specifications for the Agent Hive to autonomously fix critical Mobile PWA Dashboard runtime errors and restore full production monitoring capabilities.

## Executive Summary

The Mobile PWA Dashboard currently suffers from several critical runtime issues preventing production monitoring:

1. **Lit Component Runtime Errors**: Class field shadowing and property binding issues
2. **API Integration Failures**: Backend connectivity and real-time data flow problems  
3. **PWA Functionality Gaps**: Service worker and offline capability issues
4. **Performance Degradation**: Frame rate drops and memory usage reporting failures

This work package provides autonomous execution specifications to restore full functionality.

---

## Work Package 1: Lit Component Runtime Error Fixes

### Issue Analysis
- **Class Field Shadowing**: `@state() declare private` pattern causes runtime conflicts
- **Property Binding Errors**: Missing reactive property declarations in base classes
- **Component Lifecycle Issues**: Improper initialization order causing undefined states

### Autonomous Fix Specifications

#### File: `/src/app.ts` - Line 23-31
```typescript
// CURRENT (BROKEN):
@state() declare private currentRoute: string
@state() declare private isAuthenticated: boolean
// ... other declare statements

// FIX TO:
@state() private currentRoute: string = '/'
@state() private isAuthenticated: boolean = false
@state() private isLoading: boolean = true
@state() private isOnline: boolean = navigator.onLine
@state() private hasError: boolean = false
@state() private errorMessage: string = ''
@state() private isMobile: boolean = window.innerWidth < 768
@state() private sidebarCollapsed: boolean = false
@state() private mobileMenuOpen: boolean = false
```

#### File: `/src/views/dashboard-view.ts` - Line 25-41
```typescript
// CURRENT (BROKEN):
@state() private declare tasks: Task[]
@state() private declare agents: AgentStatus[]
// ... other declare statements

// FIX TO:
@state() private tasks: Task[] = []
@state() private agents: AgentStatus[] = []
@state() private events: TimelineEvent[] = []
@state() private systemHealth: SystemHealth | null = null
@state() private performanceMetrics: PerformanceSnapshot | null = null
@state() private healthSummary: HealthSummary | null = null
@state() private isLoading: boolean = true
@state() private error: string = ''
@state() private lastSync: Date | null = null
@state() private selectedView: 'overview' | 'kanban' | 'agents' | 'events' = 'overview'
@state() private servicesInitialized: boolean = false
@state() private wsConnected: boolean = false
@state() private realtimeEnabled: boolean = true
@state() private connectionQuality: 'excellent' | 'good' | 'poor' | 'offline' = 'offline'
@state() private updateQueue: any[] = []
@state() private lastUpdateTimestamp: Date | null = null
```

#### File: `/src/components/layout/sidebar-navigation.ts` - Line 19
```typescript
// CURRENT (BROKEN):
@state() declare private expandedItems: Set<string>

// FIX TO:
@state() private expandedItems: Set<string> = new Set()
```

### Testing Specifications
1. **Component Initialization Test**: Verify all components render without console errors
2. **Property Binding Test**: Confirm reactive updates work correctly
3. **State Management Test**: Validate state changes trigger re-renders

### Execution Time Estimate: 2 hours

---

## Work Package 2: API Integration Restoration

### Issue Analysis
- **WebSocket Connection Failures**: Incorrect URL construction in development vs production
- **Backend Adapter Integration**: Missing error handling and fallback mechanisms
- **Real-time Data Flow Interruption**: Event handling and state synchronization issues

### Autonomous Fix Specifications

#### File: `/src/services/websocket.ts` - Line 97-100
```typescript
// CURRENT (PROBLEMATIC):
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const host = window.location.hostname
const port = process.env.NODE_ENV === 'development' ? ':8000' : ''
const wsUrl = `${protocol}//${host}${port}/api/v1/ws/observability`

// FIX TO:
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const host = window.location.hostname
const port = process.env.NODE_ENV === 'development' ? ':8000' : 
            window.location.port ? `:${window.location.port}` : ''
const wsUrl = `${protocol}//${host}${port}/api/v1/ws/observability`

// Add connection validation
try {
  // Test connection availability before attempting WebSocket
  const healthCheck = await fetch(`${window.location.protocol}//${host}${port}/api/v1/health`)
  if (!healthCheck.ok) {
    throw new Error('Backend service not available')
  }
} catch (error) {
  console.warn('Backend health check failed, WebSocket connection may fail:', error)
}
```

#### File: `/src/services/backend-adapter.ts` - Enhanced Error Handling
```typescript
// ADD ERROR RECOVERY MECHANISM:
private async withRetry<T>(operation: () => Promise<T>, retries: number = 3): Promise<T> {
  for (let i = 0; i < retries; i++) {
    try {
      return await operation()
    } catch (error) {
      if (i === retries - 1) throw error
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)))
    }
  }
  throw new Error('Max retries exceeded')
}

// ENHANCE getTasksFromLiveData:
async getTasksFromLiveData(): Promise<Task[]> {
  return this.withRetry(async () => {
    const response = await fetch('/api/v1/tasks', {
      headers: this.getAuthHeaders()
    })
    
    if (!response.ok) {
      if (response.status === 401) {
        this.emit('authError', 'Authentication required')
        throw new Error('Authentication failed')
      }
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    return await response.json()
  })
}
```

### API Endpoint Validation Specifications

#### Create: `/mobile-pwa/scripts/validate_api_endpoints.js`
```javascript
const endpoints = [
  '/api/v1/health',
  '/api/v1/tasks',
  '/api/v1/agents',
  '/api/v1/system/health',
  '/api/v1/events',
  '/api/v1/metrics/performance',
  '/api/v1/ws/observability'
]

async function validateEndpoints() {
  const results = []
  
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint)
      results.push({
        endpoint,
        status: response.status,
        available: response.ok
      })
    } catch (error) {
      results.push({
        endpoint,
        status: 'ERROR',
        available: false,
        error: error.message
      })
    }
  }
  
  return results
}

// Export for testing framework
if (typeof module !== 'undefined') {
  module.exports = { validateEndpoints }
}
```

### Testing Specifications
1. **API Connectivity Test**: Validate all backend endpoints respond correctly
2. **WebSocket Connection Test**: Confirm real-time connection establishment
3. **Data Synchronization Test**: Verify offline/online state transitions
4. **Error Recovery Test**: Test retry mechanisms and fallback behaviors

### Execution Time Estimate: 4 hours

---

## Work Package 3: PWA Functionality Completion

### Issue Analysis
- **Service Worker Registration Disabled**: Development mode skips service worker entirely
- **Offline Capability Missing**: No proper caching or offline data access
- **Push Notification System Incomplete**: Firebase integration hanging in development

### Autonomous Fix Specifications

#### File: `/src/main.ts` - Line 124-128
```typescript
// CURRENT (BROKEN):
// Skip service worker registration completely in development mode
if (process.env.NODE_ENV === 'development') {
  console.log('🔧 Development mode: Skipping service worker registration entirely')
  return
}

// FIX TO:
// Allow service worker in development with proper fallbacks
if (process.env.NODE_ENV === 'development') {
  console.log('🔧 Development mode: Using service worker with development optimizations')
  
  // Use development-specific service worker
  const registration = await navigator.serviceWorker.register('/sw.dev.js', {
    scope: '/',
    updateViaCache: 'none'
  })
  
  console.log('✅ Development service worker registered:', registration)
  return registration
}
```

#### Create: `/mobile-pwa/public/sw.dev.js` - Development Service Worker
```javascript
// Development Service Worker - Minimal caching with debugging
const CACHE_NAME = 'agent-hive-dev-v1'
const urlsToCache = [
  '/',
  '/src/main.ts',
  '/src/app.ts',
  '/offline.html'
]

self.addEventListener('install', event => {
  console.log('🔧 DEV SW: Installing service worker')
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('🔧 DEV SW: Caching app shell')
        return cache.addAll(urlsToCache)
      })
      .catch(error => {
        console.warn('🔧 DEV SW: Cache add failed:', error)
      })
  )
})

self.addEventListener('fetch', event => {
  // Only cache GET requests
  if (event.request.method !== 'GET') return
  
  // Skip API requests in development
  if (event.request.url.includes('/api/')) return
  
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          console.log('🔧 DEV SW: Serving from cache:', event.request.url)
          return response
        }
        
        return fetch(event.request)
          .catch(() => {
            // Fallback to offline page
            return caches.match('/offline.html')
          })
      })
  )
})
```

#### File: `/src/services/notification.ts` - Line 92-98
```typescript
// CURRENT (HANGING IN DEV):
// In development mode, skip Firebase initialization to avoid hanging
if (process.env.NODE_ENV === 'development' && this.fcmConfig.apiKey === 'demo-key') {
  console.log('🔧 Development mode: Skipping Firebase Cloud Messaging initialization')
} else {
  // Initialize Firebase Cloud Messaging
  await this.initializeFirebaseMessaging()
}

// FIX TO:
// Use mock notification service in development
if (process.env.NODE_ENV === 'development') {
  console.log('🔧 Development mode: Using mock notification service')
  await this.initializeMockNotificationService()
} else {
  // Initialize Firebase Cloud Messaging for production
  await this.initializeFirebaseMessaging()
}

// ADD MOCK SERVICE:
private async initializeMockNotificationService(): Promise<void> {
  console.log('🔧 Initializing mock notification service for development')
  
  // Mock Firebase messaging behavior
  this.messaging = {
    getToken: () => Promise.resolve('dev-token-' + Date.now()),
    onMessage: (callback) => {
      // Simulate periodic notifications in development
      setInterval(() => {
        callback({
          notification: {
            title: '🔧 Dev Notification',
            body: 'Mock notification for testing',
            icon: '/icons/icon-192x192.png'
          },
          data: { type: 'development_test' }
        })
      }, 60000) // Every minute
    }
  }
  
  this.isInitialized = true
}
```

### PWA Manifest Validation

#### File: `/mobile-pwa/public/manifest.json` - Enhancement
```json
{
  "name": "Agent Hive - Autonomous Development Platform",
  "short_name": "Agent Hive",
  "description": "Multi-agent autonomous development monitoring dashboard",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1e293b",
  "theme_color": "#3b82f6",
  "icons": [
    {
      "src": "/icons/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ],
  "categories": ["development", "productivity", "utilities"],
  "screenshots": [
    {
      "src": "/screenshots/desktop-dashboard.png",
      "sizes": "1280x720",
      "type": "image/png",
      "form_factor": "wide",
      "label": "Main dashboard view"
    },
    {
      "src": "/screenshots/mobile-dashboard.png",
      "sizes": "390x844",
      "type": "image/png",
      "form_factor": "narrow",
      "label": "Mobile dashboard"
    }
  ],
  "orientation": "any",
  "scope": "/",
  "id": "agent-hive-pwa",
  "launch_handler": {
    "client_mode": "navigate-existing"
  },
  "edge_side_panel": {
    "preferred_width": 400
  }
}
```

### Testing Specifications
1. **PWA Installation Test**: Validate app can be installed on various devices
2. **Offline Functionality Test**: Confirm app works without network connectivity
3. **Service Worker Test**: Verify caching and background sync capabilities
4. **Notification Test**: Test push notifications and in-app notifications

### Execution Time Estimate: 3 hours

---

## Work Package 4: Performance Optimization

### Issue Analysis
- **Frame Rate Drops**: Excessive DOM updates and inefficient rendering
- **Memory Usage Reporting Failures**: Performance metrics collection issues
- **Mobile Responsiveness Problems**: Touch targets and gesture handling

### Autonomous Fix Specifications

#### File: `/src/utils/performance.ts` - Enhanced Performance Monitor
```typescript
export class PerformanceOptimizer {
  private frameCount = 0
  private lastFrameTime = performance.now()
  private fps = 0
  private memoryUsage = { used: 0, total: 0 }
  private observers: PerformanceObserver[] = []
  
  async initialize(): Promise<void> {
    // Monitor frame rate
    this.startFrameRateMonitoring()
    
    // Monitor memory usage
    this.startMemoryMonitoring()
    
    // Monitor long tasks
    this.startLongTaskMonitoring()
    
    // Monitor resource loading
    this.startResourceMonitoring()
    
    console.log('✅ Performance optimizer initialized')
  }
  
  private startFrameRateMonitoring(): void {
    const measureFrame = (currentTime: number) => {
      this.frameCount++
      
      if (currentTime - this.lastFrameTime >= 1000) {
        this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastFrameTime))
        this.frameCount = 0
        this.lastFrameTime = currentTime
        
        // Emit FPS data
        this.emit('fps-update', { fps: this.fps })
        
        // Warn if FPS drops below 30
        if (this.fps < 30) {
          console.warn(`⚠️ Low frame rate detected: ${this.fps} FPS`)
          this.emit('performance-warning', { type: 'low-fps', value: this.fps })
        }
      }
      
      requestAnimationFrame(measureFrame)
    }
    
    requestAnimationFrame(measureFrame)
  }
  
  private startMemoryMonitoring(): void {
    const updateMemoryStats = () => {
      // @ts-ignore - performance.memory is Chrome-specific
      const memory = (performance as any).memory
      if (memory) {
        this.memoryUsage = {
          used: Math.round(memory.usedJSHeapSize / 1048576), // MB
          total: Math.round(memory.totalJSHeapSize / 1048576) // MB
        }
        
        this.emit('memory-update', this.memoryUsage)
        
        // Warn if memory usage exceeds 100MB
        if (this.memoryUsage.used > 100) {
          console.warn(`⚠️ High memory usage: ${this.memoryUsage.used}MB`)
          this.emit('performance-warning', { 
            type: 'high-memory', 
            value: this.memoryUsage.used 
          })
        }
      }
    }
    
    // Update every 5 seconds
    setInterval(updateMemoryStats, 5000)
    updateMemoryStats()
  }
  
  private startLongTaskMonitoring(): void {
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > 50) { // Tasks longer than 50ms
            console.warn(`⚠️ Long task detected: ${entry.duration.toFixed(2)}ms`)
            this.emit('performance-warning', { 
              type: 'long-task', 
              value: entry.duration,
              name: entry.name 
            })
          }
        }
      })
      
      try {
        observer.observe({ entryTypes: ['longtask'] })
        this.observers.push(observer)
      } catch (error) {
        console.warn('Long task monitoring not supported')
      }
    }
  }
  
  getFPS(): number {
    return this.fps
  }
  
  getMemoryUsage(): { used: number; total: number } {
    return this.memoryUsage
  }
}
```

#### File: `/src/views/dashboard-view.ts` - Render Optimization
```typescript
// ADD EFFICIENT RENDERING METHOD:
private shouldUpdate(changedProperties: PropertyValues): boolean {
  // Only update if relevant properties changed
  const relevantProps = [
    'tasks', 'agents', 'events', 'systemHealth', 
    'performanceMetrics', 'selectedView', 'isLoading', 'error'
  ]
  
  return relevantProps.some(prop => changedProperties.has(prop as keyof this))
}

// OPTIMIZE REAL-TIME UPDATES:
private handleRealtimeTaskUpdate(taskData: Task) {
  // Batch updates to avoid excessive re-renders
  if (!this.updateBatch) {
    this.updateBatch = []
    
    // Process batch after current event loop
    setTimeout(() => {
      this.processBatchUpdates()
      this.updateBatch = null
    }, 0)
  }
  
  this.updateBatch.push({ type: 'task-update', data: taskData })
}

private updateBatch: any[] | null = null

private processBatchUpdates(): void {
  if (!this.updateBatch || this.updateBatch.length === 0) return
  
  const taskUpdates = this.updateBatch.filter(update => update.type === 'task-update')
  
  if (taskUpdates.length > 0) {
    const updatedTasks = [...this.tasks]
    
    taskUpdates.forEach(update => {
      const index = updatedTasks.findIndex(t => t.id === update.data.id)
      if (index >= 0) {
        updatedTasks[index] = { ...update.data, syncStatus: 'synced' }
      }
    })
    
    this.tasks = updatedTasks
    this.lastUpdateTimestamp = new Date()
  }
}
```

### Mobile Touch Optimizations

#### File: `/src/components/kanban/task-card.ts` - Touch Enhancements
```typescript
// ADD TOUCH-FRIENDLY STYLES:
static styles = css`
  /* Existing styles ... */
  
  .task-card {
    /* Ensure minimum touch target size */
    min-height: 44px;
    touch-action: manipulation;
  }
  
  /* Enhanced mobile interactions */
  @media (max-width: 768px) {
    .task-card {
      padding: 1rem;
      min-height: 56px; /* Larger touch targets on mobile */
    }
    
    .task-card:active {
      transform: scale(0.98);
      transition: transform 0.1s ease;
    }
  }
  
  /* Improved accessibility for touch devices */
  @media (pointer: coarse) {
    .priority-badge {
      padding: 0.25rem 0.5rem;
      font-size: 0.875rem;
    }
    
    .task-meta {
      font-size: 0.8125rem;
      gap: 0.75rem;
    }
  }
`
```

### Testing Specifications
1. **Frame Rate Test**: Monitor FPS during heavy dashboard usage
2. **Memory Usage Test**: Validate memory consumption stays under 100MB
3. **Touch Interaction Test**: Verify all interactive elements have proper touch targets
4. **Performance Regression Test**: Compare before/after optimization metrics

### Execution Time Estimate: 6 hours

---

## Autonomous Execution Framework

### Agent Coordination Specifications

#### Primary Agent Roles
1. **Frontend Specialist Agent**: Handles Lit component fixes and UI optimization
2. **Backend Integration Agent**: Manages API connectivity and WebSocket fixes  
3. **PWA Implementation Agent**: Focuses on service worker and offline capabilities
4. **Performance Optimization Agent**: Monitors and improves system performance
5. **Quality Assurance Agent**: Runs comprehensive testing and validation

#### Agent Coordination Protocol
```typescript
interface WorkPackageExecution {
  packageId: string
  assignedAgent: string
  status: 'pending' | 'in-progress' | 'completed' | 'failed'
  startTime: Date
  estimatedCompletion: Date
  actualCompletion?: Date
  dependencies: string[]
  blockers: string[]
  testResults: TestResult[]
}

interface TestResult {
  testName: string
  status: 'pass' | 'fail' | 'warning'
  details: string
  timestamp: Date
  affectedComponents: string[]
}
```

### Quality Gates
1. **Component Initialization Gate**: All components must render without console errors
2. **API Connectivity Gate**: All backend endpoints must respond correctly  
3. **PWA Validation Gate**: Service worker and offline functionality must work
4. **Performance Gate**: FPS > 30, Memory < 100MB, Load time < 3s
5. **Mobile Responsiveness Gate**: All touch targets > 44px, gestures work properly

### Rollback Procedures
Each work package includes automatic rollback if:
- More than 2 critical tests fail
- Performance degrades by >20%
- New console errors are introduced
- User interaction becomes unavailable

### Success Criteria
- **Zero runtime console errors**
- **All API endpoints responding correctly**
- **PWA installable and functional offline**
- **60+ FPS on desktop, 30+ FPS on mobile**
- **Mobile responsiveness validated across devices**
- **Real-time dashboard updates functioning**
- **WebSocket connection stable and recovering properly**

### Monitoring and Reporting
Each agent will provide progress updates every 30 minutes including:
- Tasks completed vs. estimated
- Test results summary
- Performance metrics comparison
- Blocker identification and resolution status
- Next steps and dependencies

---

## Expected Timeline

| Work Package | Estimated Duration | Dependencies | Critical Path |
|--------------|-------------------|--------------|---------------|
| 1. Lit Component Fixes | 2 hours | None | Yes |
| 2. API Integration | 4 hours | Package 1 | Yes |
| 3. PWA Functionality | 3 hours | Package 1 | No |
| 4. Performance Optimization | 6 hours | Packages 1-2 | No |

**Total Estimated Time**: 8-10 hours (accounting for parallel execution)
**Critical Path Duration**: 6 hours

### Execution Strategy
1. **Phase 1** (0-2h): Execute Package 1 (Lit Component Fixes)
2. **Phase 2** (2-6h): Execute Package 2 (API Integration) while starting Package 3 (PWA) in parallel
3. **Phase 3** (4-10h): Execute Package 4 (Performance) while finalizing Package 3
4. **Phase 4** (8-10h): Integration testing, validation, and deployment

This work package provides complete autonomous execution specifications for restoring the Mobile PWA Dashboard to full production monitoring capabilities. Each section includes detailed technical specifications, testing procedures, and quality gates to ensure successful autonomous implementation by the Agent Hive system.
import './styles/main.css'
import './app'
import { AuthService } from './services/auth'
import { WebSocketService } from './services/websocket'
import { NotificationService } from './services/notification'
import { OfflineService } from './services/offline'
import { PerformanceMonitor } from './utils/performance'

// Initialize services
const authService = AuthService.getInstance()
const wsService = WebSocketService.getInstance()
const notificationService = NotificationService.getInstance()
const offlineService = OfflineService.getInstance()
const perfMonitor = PerformanceMonitor.getInstance()

// App initialization
class AppInitializer {
  private static instance: AppInitializer
  
  static getInstance(): AppInitializer {
    if (!AppInitializer.instance) {
      AppInitializer.instance = new AppInitializer()
    }
    return AppInitializer.instance
  }
  
  async initialize(): Promise<void> {
    try {
      console.log('🚀 Initializing LeanVibe Agent Hive Mobile PWA...')
      
      // Start performance monitoring
      perfMonitor.startSession()
      
      // Initialize offline service first (sets up IndexedDB)
      await offlineService.initialize()
      
      // Initialize notification service
      await notificationService.initialize()
      
      // Check authentication state
      await authService.initialize()
      
      // Initialize WebSocket if authenticated
      if (authService.isAuthenticated()) {
        await wsService.initialize()
      }
      
      // Mark app as ready
      document.body.classList.add('app-ready')
      
      console.log('✅ App initialization complete')
      
      // Track initialization time
      perfMonitor.track('app_initialization', performance.now())
      
    } catch (error) {
      console.error('❌ App initialization failed:', error)
      this.handleInitializationError(error)
    }
  }
  
  private handleInitializationError(error: unknown): void {
    // Show error UI
    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
    
    const errorContainer = document.createElement('div')
    errorContainer.className = 'fixed inset-0 bg-red-50 flex items-center justify-center p-4 z-50'
    errorContainer.innerHTML = `
      <div class="bg-white p-6 rounded-lg shadow-lg max-w-md w-full">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <svg class="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">Initialization Error</h3>
            <div class="mt-2 text-sm text-red-700">
              <p>${errorMessage}</p>
            </div>
            <div class="mt-4">
              <button 
                onclick="location.reload()" 
                class="bg-red-600 text-white px-4 py-2 rounded text-sm hover:bg-red-700"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    `
    
    document.body.appendChild(errorContainer)
  }
}

// Start the app
const app = AppInitializer.getInstance()
app.initialize()

// Handle online/offline events
window.addEventListener('online', () => {
  console.log('🌐 App is online')
  if (authService.isAuthenticated()) {
    wsService.reconnect()
  }
})

window.addEventListener('offline', () => {
  console.log('📱 App is offline')
})

// Handle visibility changes (for mobile app lifecycle)
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    console.log('👁️ App is visible')
    perfMonitor.track('app_visible', performance.now())
    if (authService.isAuthenticated()) {
      wsService.reconnect()
    }
  } else {
    console.log('🙈 App is hidden')
    perfMonitor.track('app_hidden', performance.now())
  }
})

// Performance monitoring
window.addEventListener('load', () => {
  // Report Core Web Vitals
  if ('web-vital' in window) {
    // @ts-ignore
    import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
      getCLS(perfMonitor.reportWebVital.bind(perfMonitor))
      getFID(perfMonitor.reportWebVital.bind(perfMonitor))
      getFCP(perfMonitor.reportWebVital.bind(perfMonitor))
      getLCP(perfMonitor.reportWebVital.bind(perfMonitor))
      getTTFB(perfMonitor.reportWebVital.bind(perfMonitor))
    })
  }
})

// Global error handling
window.addEventListener('error', (event) => {
  console.error('🚨 Global error:', event.error)
  perfMonitor.reportError(event.error)
})

window.addEventListener('unhandledrejection', (event) => {
  console.error('🚨 Unhandled promise rejection:', event.reason)
  perfMonitor.reportError(event.reason)
})

// Export for debugging
if (process.env.NODE_ENV === 'development') {
  // @ts-ignore
  window.appServices = {
    auth: authService,
    websocket: wsService,
    notification: notificationService,
    offline: offlineService,
    performance: perfMonitor
  }
}
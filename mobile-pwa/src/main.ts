import './styles/main.css'
import './app'
import { AuthService } from './services/auth'
import { WebSocketService } from './services/websocket'
import { NotificationService } from './services/notification'
import { OfflineService } from './services/offline'
import { PerformanceOptimizer } from './utils/performance'
import { backendAdapter } from './services/backend-adapter'

// Initialize services
const authService = AuthService.getInstance()
const wsService = WebSocketService.getInstance()
const notificationService = NotificationService.getInstance()
const offlineService = OfflineService.getInstance()
const perfMonitor = PerformanceOptimizer.getInstance()

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
      await perfMonitor.initialize()
      
      // Initialize offline service first (sets up IndexedDB)
      await offlineService.initialize()
      
      // Skip notification service in development to avoid hanging
      if (process.env.NODE_ENV === 'production') {
        // Initialize notification service
        try {
          await notificationService.initialize()
        } catch (error) {
          console.warn('⚠️ Notification service initialization failed:', error.message)
        }
      } else {
        console.log('🔧 Development mode: Skipping notification service initialization')
      }
      
      // Check authentication state
      await authService.initialize()
      
      // Create and mount the app element first (don't wait for WebSocket)
      const appContainer = document.getElementById('app')
      if (appContainer) {
        const appElement = document.createElement('agent-hive-app')
        appContainer.appendChild(appElement)
      }
      
      // Hide loading screen
      const loadingContainer = document.querySelector('.loading-container')
      if (loadingContainer) {
        loadingContainer.style.display = 'none'
      }
      
      // Mark app as ready
      document.body.classList.add('app-ready')
      
      console.log('✅ App initialization complete')
      
      // Initialize WebSocket in background (non-blocking)
      if (authService.isAuthenticated()) {
        wsService.initialize().catch(error => {
          console.warn('⚠️ WebSocket initialization failed, continuing without real-time updates:', error)
        })
      }
      
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

// Register service worker for PWA functionality
async function registerServiceWorker(): Promise<void> {
  if ('serviceWorker' in navigator) {
    try {
      console.log('🔧 Registering service worker...')
      
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/',
        updateViaCache: 'none'
      })
      
      console.log('✅ Service worker registered:', registration)
      
      // Handle updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing
        if (newWorker) {
          console.log('🔄 New service worker available')
          
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              console.log('🔄 New service worker installed, update available')
              
              // Notify user of update
              const updateEvent = new CustomEvent('sw-update-available', {
                detail: { registration }
              })
              window.dispatchEvent(updateEvent)
            }
          })
        }
      })
      
      // Handle controller change (new SW activated)
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        console.log('🔄 Service worker controller changed, reloading...')
        window.location.reload()
      })
      
      // Send messages to service worker
      if (registration.active) {
        registration.active.postMessage({
          type: 'CLIENT_READY',
          data: { timestamp: Date.now() }
        })
      }
      
    } catch (error) {
      console.error('❌ Service worker registration failed:', error)
    }
  } else {
    console.warn('⚠️ Service workers not supported')
  }
}

// Start the app and register service worker
const appInitializer = AppInitializer.getInstance()

Promise.all([
  registerServiceWorker(),
  appInitializer.initialize()
]).then(() => {
  console.log('🎉 App and service worker ready')
}).catch(error => {
  console.error('❌ Failed to start app:', error)
})

// Handle service worker updates
window.addEventListener('sw-update-available', (event: CustomEvent) => {
  console.log('🔄 Service worker update available')
  
  // Show update notification
  const updateNotification = document.createElement('div')
  updateNotification.className = 'fixed top-4 right-4 bg-blue-600 text-white p-4 rounded-lg shadow-lg z-50 max-w-sm'
  updateNotification.innerHTML = `
    <div class="flex items-start">
      <div class="flex-shrink-0">
        <svg class="h-5 w-5 text-blue-200" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
        </svg>
      </div>
      <div class="ml-3 flex-1">
        <p class="text-sm font-medium">Update Available</p>
        <p class="mt-1 text-sm text-blue-200">A new version is ready to install.</p>
        <div class="mt-3 flex space-x-2">
          <button 
            id="update-now" 
            class="bg-blue-800 text-white px-3 py-1 rounded text-sm hover:bg-blue-900"
          >
            Update Now
          </button>
          <button 
            id="update-later" 
            class="bg-transparent text-blue-200 px-3 py-1 rounded text-sm hover:text-white border border-blue-200"
          >
            Later
          </button>
        </div>
      </div>
    </div>
  `
  
  document.body.appendChild(updateNotification)
  
  // Handle update actions
  const updateNowBtn = updateNotification.querySelector('#update-now')
  const updateLaterBtn = updateNotification.querySelector('#update-later')
  
  updateNowBtn?.addEventListener('click', () => {
    const registration = event.detail.registration
    const newWorker = registration.waiting
    
    if (newWorker) {
      newWorker.postMessage({ type: 'SKIP_WAITING' })
    }
  })
  
  updateLaterBtn?.addEventListener('click', () => {
    updateNotification.remove()
  })
  
  // Auto-hide after 10 seconds
  setTimeout(() => {
    if (updateNotification.parentNode) {
      updateNotification.remove()
    }
  }, 10000)
})

// Handle online/offline events
window.addEventListener('online', () => {
  console.log('🌐 App is online')
  if (authService.isAuthenticated()) {
    wsService.reconnect()
  }
  
  // Notify offline service about online state
  const onlineEvent = new CustomEvent('app-online')
  window.dispatchEvent(onlineEvent)
})

window.addEventListener('offline', () => {
  console.log('📱 App is offline')
  
  // Notify offline service about offline state
  const offlineEvent = new CustomEvent('app-offline')
  window.dispatchEvent(offlineEvent)
})

// Handle visibility changes (for mobile app lifecycle)
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    console.log('👁️ App is visible')
    console.log('App visible')
    if (authService.isAuthenticated()) {
      wsService.reconnect()
    }
  } else {
    console.log('🙈 App is hidden')
    console.log('App hidden')
  }
})

// Performance monitoring
window.addEventListener('load', () => {
  console.log('🏁 App fully loaded')
  // Web vitals tracking disabled due to missing dependency
})

// Global error handling
window.addEventListener('error', (event) => {
  console.error('🚨 Global error:', event.error)
  // Error reporting disabled for now
})

window.addEventListener('unhandledrejection', (event) => {
  console.error('🚨 Unhandled promise rejection:', event.reason)
  // Error reporting disabled for now
})

// Export for debugging
if (process.env.NODE_ENV === 'development') {
  // @ts-ignore
  window.appServices = {
    auth: authService,
    websocket: wsService,
    notification: notificationService,
    offline: offlineService,
    performance: perfMonitor,
    backendAdapter: backendAdapter
  }
}
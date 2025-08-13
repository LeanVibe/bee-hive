import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { Router } from './router/router'
import { AuthService } from './services/auth'
import { WebSocketService } from './services/websocket'
import { NotificationService } from './services/notification'
import { OfflineService } from './services/offline'
import './components/layout/app-header'
import './components/layout/bottom-navigation'
import './components/layout/sidebar-navigation'
import './components/layout/install-prompt'
import './components/common/error-boundary'
import './components/common/loading-spinner'
import './views/dashboard-view'
import './views/mobile-enhanced-dashboard-view'
import './views/agents-view'
import './views/tasks-view'
import './views/system-health-view'
import './views/performance-analytics-view'
import './views/login-view'
import './components/autonomous-development-dashboard'

@customElement('agent-hive-app')
export class AgentHiveApp extends LitElement {
  @state() declare private currentRoute: string
  @state() declare private isAuthenticated: boolean
  @state() declare private isLoading: boolean
  @state() declare private isOnline: boolean
  @state() declare private hasError: boolean
  @state() declare private errorMessage: string
  @state() declare private isMobile: boolean
  @state() declare private sidebarCollapsed: boolean
  @state() declare private mobileMenuOpen: boolean
  
  private router: Router
  private authService: AuthService
  private wsService: WebSocketService
  private notificationService: NotificationService
  private offlineService: OfflineService
  
  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .app-container {
      display: flex;
      height: 100%;
      position: relative;
    }

    /* Desktop Layout */
    .app-container.desktop {
      flex-direction: row;
    }

    /* Mobile Layout */
    .app-container.mobile {
      flex-direction: column;
    }
    
    .sidebar-container {
      flex-shrink: 0;
      z-index: 30;
    }

    .main-layout {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      min-width: 0;
    }
    
    .main-content {
      flex: 1;
      overflow: hidden;
      position: relative;
    }
    
    .view-container {
      height: 100%;
      overflow-y: auto;
      overflow-x: hidden;
      -webkit-overflow-scrolling: touch;
      scroll-behavior: smooth;
    }

    /* Mobile Menu Overlay */
    .mobile-menu-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      backdrop-filter: blur(4px);
      z-index: 25;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }

    .mobile-menu-overlay.open {
      opacity: 1;
      pointer-events: all;
    }

    /* Header adjustments for responsive */
    .header-container {
      display: none;
    }

    .app-container.mobile .header-container {
      display: block;
    }
    
    .offline-banner {
      background: linear-gradient(90deg, #f59e0b, #f97316);
      color: white;
      padding: 0.5rem 1rem;
      text-align: center;
      font-size: 0.875rem;
      font-weight: 500;
      animation: slideDown 0.3s ease-out;
    }
    
    .error-banner {
      background: linear-gradient(90deg, #dc2626, #b91c1c);
      color: white;
      padding: 0.75rem 1rem;
      text-align: center;
      font-size: 0.875rem;
      font-weight: 500;
      animation: slideDown 0.3s ease-out;
    }
    
    .loading-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(248, 250, 252, 0.9);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }
    
    .fade-in {
      animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes slideDown {
      from {
        transform: translateY(-100%);
        opacity: 0;
      }
      to {
        transform: translateY(0);
        opacity: 1;
      }
    }
    
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    /* Safe area support for devices with notches */
    @supports (padding: max(0px)) {
      .app-container {
        padding-top: env(safe-area-inset-top);
        padding-bottom: env(safe-area-inset-bottom);
      }
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
      .offline-banner,
      .error-banner {
        border: 2px solid currentColor;
      }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      :host {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      }
      
      .loading-overlay {
        background: rgba(15, 23, 42, 0.9);
      }
    }
  `
  
  constructor() {
    super()
    
    // Initialize state properties
    this.currentRoute = '/'
    this.isAuthenticated = false
    this.isLoading = true
    this.isOnline = navigator.onLine
    this.hasError = false
    this.errorMessage = ''
    this.isMobile = window.innerWidth < 768
    this.sidebarCollapsed = false
    this.mobileMenuOpen = false
    
    // Initialize services
    this.authService = AuthService.getInstance()
    this.wsService = WebSocketService.getInstance()
    this.notificationService = NotificationService.getInstance()
    this.offlineService = OfflineService.getInstance()
    
    // Initialize router
    this.router = new Router()
    this.setupRoutes()
    
    // Listen to service events
    this.setupEventListeners()
  }
  
  connectedCallback() {
    super.connectedCallback()
    this.initializeApp()
    this.setupResponsiveHandlers()
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    this.removeEventListeners()
  }
  
  private async initializeApp() {
    try {
      this.isLoading = true
      
      // Initialize authentication service first
      await this.authService.initialize()
      // Initialize offline service
      await this.offlineService.initialize()
      
      // Check authentication state
      this.isAuthenticated = this.authService.isAuthenticated()
      
      // Start router AFTER auth initialization
      this.router.start()
      this.currentRoute = this.router.getCurrentRoute()
      
      // If authenticated, ensure WebSocket is connected
      if (this.isAuthenticated) {
        await this.wsService.ensureConnection()
      }
      
      this.isLoading = false
      this.hasError = false
      
    } catch (error) {
      console.error('App initialization error:', error)
      this.hasError = true
      this.errorMessage = error instanceof Error ? error.message : 'Failed to initialize app'
      this.isLoading = false
    }
  }
  
  private setupRoutes() {
    // Public routes
    this.router.addRoute('/login', () => this.setRoute('/login'))
    
    // Protected routes
    this.router.addRoute('/', () => this.setRoute('/'), { requireAuth: true })
    this.router.addRoute('/dashboard', () => this.setRoute('/dashboard'), { requireAuth: true })
    this.router.addRoute('/autonomous', () => this.setRoute('/autonomous'), { requireAuth: true })
    this.router.addRoute('/tasks', () => this.setRoute('/tasks'), { requireAuth: true })
    this.router.addRoute('/agents', () => this.setRoute('/agents'), { requireAuth: true })
    this.router.addRoute('/system-health', () => this.setRoute('/system-health'), { requireAuth: true })
    this.router.addRoute('/performance', () => this.setRoute('/performance'), { requireAuth: true })
    this.router.addRoute('/events', () => this.setRoute('/events'), { requireAuth: true })
    this.router.addRoute('/settings', () => this.setRoute('/settings'), { requireAuth: true })
    
    // Handle route changes
    this.router.onRouteChange((route) => {
      this.setRoute(route)
    })
  }
  
  private setRoute(route: string) {
    this.currentRoute = route
    this.requestUpdate()
  }
  
  private setupEventListeners() {
    // Authentication events
    this.authService.on('authenticated', () => {
      this.isAuthenticated = true
      this.wsService.ensureConnection()
      this.router.navigate('/dashboard')
    })
    
    this.authService.on('unauthenticated', () => {
      this.isAuthenticated = false
      this.wsService.disconnect()
      this.router.navigate('/login')
    })
    this.authService.on('auth-expired', () => {
      this.isAuthenticated = false
      this.wsService.disconnect()
      this.showError('Session expired. Please sign in again.')
      this.router.navigate('/login')
    })
    
    // Network events
    window.addEventListener('online', this.handleOnline.bind(this))
    window.addEventListener('offline', this.handleOffline.bind(this))
    
    // WebSocket events
    this.wsService.on('connected', () => {
      this.hasError = false
    })
    
    this.wsService.on('disconnected', () => {
      if (this.isAuthenticated && this.isOnline) {
        this.showError('Connection lost. Attempting to reconnect...')
      }
    })
    
    this.wsService.on('error', (error: Error) => {
      this.showError(`WebSocket error: ${error.message}`)
    })

    // WS contract governance
    this.wsService.on('version-mismatch', (data: { current: string, supported: string[] }) => {
      this.showError(`Version mismatch: server ${data.current} not in supported ${data.supported.join(', ')}`)
    })
    
    // Notification events
    this.notificationService.on('notification', (notification) => {
      this.showNotification(notification)
    })
  }
  
  private removeEventListeners() {
    window.removeEventListener('online', this.handleOnline.bind(this))
    window.removeEventListener('offline', this.handleOffline.bind(this))
    window.removeEventListener('resize', this.handleResize.bind(this))
  }

  private setupResponsiveHandlers() {
    // Listen for window resize
    window.addEventListener('resize', this.handleResize.bind(this))
    
    // Check initial responsive state
    this.handleResize()
  }

  private handleResize() {
    const wasMobile = this.isMobile
    this.isMobile = window.innerWidth < 768
    
    // Close mobile menu when switching to desktop
    if (wasMobile && !this.isMobile) {
      this.mobileMenuOpen = false
    }
  }

  private handleSidebarToggle(event: CustomEvent) {
    this.sidebarCollapsed = event.detail.collapsed
  }

  private handleMobileMenuToggle() {
    this.mobileMenuOpen = !this.mobileMenuOpen
  }

  private closeMobileMenu() {
    this.mobileMenuOpen = false
  }
  
  private handleOnline() {
    this.isOnline = true
    if (this.isAuthenticated) {
      this.wsService.ensureConnection()
    }
  }
  
  private handleOffline() {
    this.isOnline = false
    this.showError('You are offline. Changes will sync when you are back online.')
  }
  
  private showError(message: string) {
    this.hasError = true
    this.errorMessage = message
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
      this.hasError = false
    }, 5000)
  }
  
  private showNotification(notification: any) {
    // Handle notifications (could show toast, etc.)
    console.log('Notification:', notification)
  }
  
  private renderCurrentView() {
    if (!this.isAuthenticated && this.currentRoute !== '/login') {
      return html`<login-view></login-view>`
    }
    
    switch (this.currentRoute) {
      case '/login':
        return html`<login-view></login-view>`
      case '/':
      case '/dashboard':
        return this.isMobile ? 
          html`<mobile-enhanced-dashboard-view .mobile=${true} .decisionMode=${true}></mobile-enhanced-dashboard-view>` :
          html`<dashboard-view></dashboard-view>`
      case '/autonomous':
        return html`<autonomous-development-dashboard></autonomous-development-dashboard>`
      case '/tasks':
        return html`<tasks-view></tasks-view>`
      case '/agents':
        return html`<agents-view></agents-view>`
      case '/system-health':
        return html`<system-health-view></system-health-view>`
      case '/performance':
        return html`<performance-analytics-view></performance-analytics-view>`
      case '/events':
      case '/settings':
        return html`<div style="padding: 2rem; text-align: center; color: #6b7280;">
          <h2 style="font-size: 1.25rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">
            Coming Soon
          </h2>
          <p style="margin-bottom: 1.5rem;">This feature is under development.</p>
          <button 
            @click="${() => this.router.navigate('/dashboard')}"
            style="background: #3b82f6; color: white; padding: 0.5rem 1rem; border: none; border-radius: 0.375rem; cursor: pointer; font-weight: 500;"
          >
            Back to Dashboard
          </button>
        </div>`
      default:
        return html`<div style="padding: 2rem; text-center; color: #6b7280;">
          <h2 style="font-size: 1.25rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">
            Page not found
          </h2>
          <p style="margin-bottom: 1.5rem;">The requested page could not be found.</p>
          <button 
            @click="${() => this.router.navigate('/dashboard')}"
            style="background: #3b82f6; color: white; padding: 0.5rem 1rem; border: none; border-radius: 0.375rem; cursor: pointer; font-weight: 500;"
          >
            Go to Dashboard
          </button>
        </div>`
    }
  }
  
  render() {
    return html`
      <div class="app-container ${this.isMobile ? 'mobile' : 'desktop'}">
        <!-- Error banner -->
        ${this.hasError ? html`
          <div class="error-banner">
            <span>${this.errorMessage}</span>
            <button 
              @click="${() => this.hasError = false}"
              class="ml-2 text-white hover:text-gray-200"
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        ` : ''}
        
        <!-- Offline banner -->
        ${!this.isOnline ? html`
          <div class="offline-banner">
            📱 You're offline. Some features may be limited.
          </div>
        ` : ''}

        <!-- Mobile Menu Overlay -->
        ${this.isMobile ? html`
          <div 
            class="mobile-menu-overlay ${this.mobileMenuOpen ? 'open' : ''}"
            @click="${this.closeMobileMenu}"
          ></div>
        ` : ''}
        
        <!-- Sidebar Navigation (Desktop) or Mobile Menu -->
        ${this.isAuthenticated && this.currentRoute !== '/login' ? html`
          <div class="sidebar-container">
            <sidebar-navigation
              .currentRoute="${this.currentRoute}"
              .collapsed="${this.sidebarCollapsed}"
              .mobile="${this.isMobile}"
              .open="${this.mobileMenuOpen}"
              @navigate="${(e: CustomEvent) => {
                this.router.navigate(e.detail.route)
                this.closeMobileMenu()
              }}"
              @sidebar-toggle="${this.handleSidebarToggle}"
            ></sidebar-navigation>
          </div>
        ` : ''}

        <!-- Main Layout -->
        <div class="main-layout">
          <!-- Header (Mobile only) -->
          <div class="header-container">
            ${this.isAuthenticated && this.isMobile ? html`
              <app-header 
                .currentRoute="${this.currentRoute}"
                .isOnline="${this.isOnline}"
                @menu-toggle="${this.handleMobileMenuToggle}"
              ></app-header>
            ` : ''}
          </div>
          
          <!-- Main content -->
          <main class="main-content">
            <div class="view-container">
              <error-boundary>
                ${this.renderCurrentView()}
              </error-boundary>
            </div>
          </main>
          
          <!-- Bottom navigation (Mobile only) -->
          ${this.isAuthenticated && this.isMobile && this.currentRoute !== '/login' ? html`
            <bottom-navigation 
              .currentRoute="${this.currentRoute}"
              @navigate="${(e: CustomEvent) => this.router.navigate(e.detail.route)}"
            ></bottom-navigation>
          ` : ''}
        </div>
        
        <!-- Install prompt -->
        <install-prompt></install-prompt>
        
        <!-- Loading overlay -->
        ${this.isLoading ? html`
          <div class="loading-overlay">
            <loading-spinner size="large"></loading-spinner>
          </div>
        ` : ''}
      </div>
    `
  }
}
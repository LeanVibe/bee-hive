import { AuthService } from '../services/auth'
import { AuthGuard, RouteGuardOptions } from '../services/auth-guard'

export interface Route {
  path: string
  handler: () => void
  options?: RouteGuardOptions & {
    redirect?: string
    title?: string
    meta?: Record<string, any>
  }
}

export class Router {
  private routes: Map<string, Route> = new Map()
  private currentRoute: string = '/'
  private authService: AuthService
  private authGuard: AuthGuard
  private isStarted: boolean = false
  
  constructor() {
    this.authService = AuthService.getInstance()
    this.authGuard = AuthGuard.getInstance()
    this.setupAuthEventHandlers()
  }
  
  addRoute(path: string, handler: () => void, options?: Route['options']): void {
    this.routes.set(path, { path, handler, options })
  }
  
  start(): void {
    if (this.isStarted) return
    
    // Handle browser navigation
    window.addEventListener('popstate', this.handlePopState.bind(this))
    
    // Handle initial route
    this.handleRoute(window.location.pathname)
    
    this.isStarted = true
    console.log('🛣️ Router started')
  }
  
  stop(): void {
    if (!this.isStarted) return
    
    window.removeEventListener('popstate', this.handlePopState.bind(this))
    this.isStarted = false
  }
  
  
  replace(path: string): void {
    this.navigate(path, true)
  }
  
  back(): void {
    window.history.back()
  }
  
  forward(): void {
    window.history.forward()
  }
  
  private handlePopState(event: PopStateEvent): void {
    this.handleRoute(window.location.pathname)
  }
  
  private async handleRoute(path: string): Promise<void> {
    console.log('🛣️ Navigating to:', path)
    
    // Normalize path
    const normalizedPath = this.normalizePath(path)
    
    // Find matching route
    const route = this.findRoute(normalizedPath)
    
    if (!route) {
      console.warn('Route not found:', normalizedPath)
      this.handleNotFound(normalizedPath)
      return
    }
    
    // Use AuthGuard for comprehensive security checking
    if (route.options) {
      const guardResult = await this.authGuard.canActivate(normalizedPath, route.options)
      
      if (!guardResult.allowed) {
        console.warn('Access denied:', guardResult.reason)
        
        if (guardResult.redirectTo) {
          this.navigate(guardResult.redirectTo, true)
        } else {
          this.handleAccessDenied(normalizedPath, guardResult.reason)
        }
        return
      }
    }
    
    // Handle redirect
    if (route.options?.redirect) {
      this.navigate(route.options.redirect, true)
      return
    }
    
    // Update page title if specified
    if (route.options?.title) {
      document.title = `${route.options.title} - LeanVibe Agent Hive`
    }
    
    // Update current route
    this.currentRoute = normalizedPath
    
    // Execute route handler
    try {
      route.handler()
      this.notifyRouteChange(normalizedPath)
    } catch (error) {
      console.error('Route handler error:', error)
      this.handleRouteError(normalizedPath, error)
    }
  }
  
  private normalizePath(path: string): string {
    // Remove trailing slash unless it's the root
    if (path.length > 1 && path.endsWith('/')) {
      path = path.slice(0, -1)
    }
    
    // Ensure path starts with /
    if (!path.startsWith('/')) {
      path = '/' + path
    }
    
    return path
  }
  
  private findRoute(path: string): Route | undefined {
    // Try exact match first
    const exactMatch = this.routes.get(path)
    if (exactMatch) return exactMatch
    
    // Try pattern matching for dynamic routes
    for (const [routePath, route] of this.routes) {
      if (this.matchesPattern(path, routePath)) {
        return route
      }
    }
    
    return undefined
  }
  
  private matchesPattern(path: string, pattern: string): boolean {
    // Simple pattern matching - could be extended for parameters
    if (pattern.includes(':')) {
      const patternParts = pattern.split('/')
      const pathParts = path.split('/')
      
      if (patternParts.length !== pathParts.length) return false
      
      for (let i = 0; i < patternParts.length; i++) {
        const patternPart = patternParts[i]
        const pathPart = pathParts[i]
        
        if (patternPart.startsWith(':')) {
          // Parameter - matches any value
          continue
        } else if (patternPart !== pathPart) {
          return false
        }
      }
      
      return true
    }
    
    return false
  }
  
  private handleNotFound(path: string): void {
    console.warn('404 - Route not found:', path)
    
    // Try to redirect to a sensible default
    if (this.authService.isAuthenticated()) {
      this.navigate('/dashboard', true)
    } else {
      this.navigate('/login', true)
    }
  }
  
  private handleUnauthorized(path: string): void {
    console.warn('403 - Unauthorized access to:', path)
    
    // Redirect to dashboard for authenticated users
    if (this.authService.isAuthenticated()) {
      this.navigate('/dashboard', true)
    } else {
      this.navigate('/login', true)
    }
  }
  
  private handleAccessDenied(path: string, reason?: string): void {
    console.warn('Access denied to:', path, reason)
    
    // Emit access denied event for UI components to handle
    window.dispatchEvent(new CustomEvent('router-access-denied', {
      detail: { path, reason }
    }))
    
    // Redirect to appropriate page
    if (this.authService.isAuthenticated()) {
      this.navigate('/access-denied', true)
    } else {
      this.navigate('/login', true)
    }
  }
  
  private setupAuthEventHandlers(): void {
    // Listen for auth redirect events from AuthGuard
    window.addEventListener('auth-redirect', (event: any) => {
      const { path, reason } = event.detail
      console.log('Auth redirect triggered:', path, reason)
      this.navigate(path, true)
    })
    
    // Listen for auth access denied events
    window.addEventListener('auth-access-denied', (event: any) => {
      const { path, reason } = event.detail
      this.handleAccessDenied(path, reason)
    })
    
    // Listen for authentication state changes
    this.authService.on('authenticated', () => {
      // If we're on login page and user just authenticated, redirect to dashboard
      if (this.currentRoute === '/login') {
        this.navigate('/dashboard', true)
      }
    })
    
    this.authService.on('unauthenticated', () => {
      // If user logged out and we're on a protected route, redirect to login
      if (this.currentRoute !== '/login' && this.currentRoute !== '/') {
        this.navigate('/login', true)
      }
    })
  }
  
  private handleRouteError(path: string, error: unknown): void {
    console.error('Route error for', path, ':', error)
    
    // Could show an error page or redirect to a safe route
    if (this.authService.isAuthenticated()) {
      this.navigate('/dashboard', true)
    } else {
      this.navigate('/login', true)
    }
  }
  
  // Event handling
  private routeChangeHandlers: ((route: string) => void)[] = []
  
  onRouteChange(handler: (route: string) => void): () => void {
    this.routeChangeHandlers.push(handler)
    
    // Return unsubscribe function
    return () => {
      const index = this.routeChangeHandlers.indexOf(handler)
      if (index > -1) {
        this.routeChangeHandlers.splice(index, 1)
      }
    }
  }
  
  private notifyRouteChange(route: string): void {
    this.routeChangeHandlers.forEach(handler => {
      try {
        handler(route)
      } catch (error) {
        console.error('Route change handler error:', error)
      }
    })
  }
  
  // Utility methods
  getCurrentRoute(): string {
    return this.currentRoute
  }
  
  isCurrentRoute(path: string): boolean {
    return this.normalizePath(path) === this.currentRoute
  }
  
  getRouteParams(path: string): Record<string, string> {
    const route = this.findRoute(this.currentRoute)
    if (!route || !route.path.includes(':')) {
      return {}
    }
    
    const patternParts = route.path.split('/')
    const pathParts = this.currentRoute.split('/')
    const params: Record<string, string> = {}
    
    for (let i = 0; i < patternParts.length; i++) {
      const patternPart = patternParts[i]
      if (patternPart.startsWith(':')) {
        const paramName = patternPart.slice(1)
        params[paramName] = pathParts[i]
      }
    }
    
    return params
  }
  
  buildPath(pattern: string, params: Record<string, string>): string {
    let path = pattern
    
    for (const [key, value] of Object.entries(params)) {
      path = path.replace(`:${key}`, value)
    }
    
    return path
  }
  
  // Guard methods for components
  async canActivate(path: string): Promise<boolean> {
    const route = this.findRoute(path)
    if (!route) return false
    
    if (route.options) {
      const guardResult = await this.authGuard.canActivate(path, route.options)
      return guardResult.allowed
    }
    
    return true
  }
  
  // Synchronous version for backwards compatibility
  canActivateSync(path: string): boolean {
    const route = this.findRoute(path)
    if (!route) return false
    
    // Basic checks only
    if (route.options?.requireAuth && !this.authService.isAuthenticated()) {
      return false
    }
    
    return true
  }
  
  // URL utilities
  getFullUrl(path: string): string {
    const baseUrl = window.location.origin
    return baseUrl + this.normalizePath(path)
  }
  
  getQueryParams(): URLSearchParams {
    return new URLSearchParams(window.location.search)
  }
  
  getQueryParam(name: string): string | null {
    return this.getQueryParams().get(name)
  }
  
  setQueryParams(params: Record<string, string>, replace: boolean = false): void {
    const url = new URL(window.location.href)
    
    for (const [key, value] of Object.entries(params)) {
      if (value === null || value === undefined) {
        url.searchParams.delete(key)
      } else {
        url.searchParams.set(key, value)
      }
    }
    
    const newUrl = url.pathname + url.search + url.hash
    
    if (replace) {
      window.history.replaceState({}, '', newUrl)
    } else {
      window.history.pushState({}, '', newUrl)
    }
  }
  
  // Hash utilities
  getHash(): string {
    return window.location.hash.slice(1) // Remove #
  }
  
  setHash(hash: string): void {
    window.location.hash = hash
  }
  
  // Preloading and prefetching
  prefetchRoute(path: string): void {
    // In a more complex router, this would preload route components
    console.log('Prefetching route:', path)
  }
  
  // Advanced guard methods
  async getSecurityContext(): Promise<any> {
    return this.authGuard.getSecurityContext()
  }
  
  async checkMultiplePermissions(checks: Array<{
    name: string
    options: RouteGuardOptions
  }>): Promise<Record<string, boolean>> {
    return this.authGuard.checkMultiplePermissions(checks)
  }
  
  canUserPerformAction(action: string, context?: any): boolean {
    return this.authGuard.canUserPerformAction(action, context)
  }
  
  // Route metadata utilities
  getRouteMetadata(path: string): Record<string, any> | undefined {
    const route = this.findRoute(path)
    return route?.options?.meta
  }
  
  setRouteMetadata(path: string, meta: Record<string, any>): void {
    const route = this.findRoute(path)
    if (route && route.options) {
      route.options.meta = { ...route.options.meta, ...meta }
    }
  }
  
  // Navigation history utilities
  private navigationHistory: string[] = []
  
  getNavigationHistory(): string[] {
    return [...this.navigationHistory]
  }
  
  canGoBack(): boolean {
    return this.navigationHistory.length > 1
  }
  
  getPreviousRoute(): string | null {
    return this.navigationHistory.length > 1 
      ? this.navigationHistory[this.navigationHistory.length - 2]
      : null
  }
  
  // Enhanced navigation with history tracking (override original navigate method)
  navigate(path: string, replace: boolean = false): void {
    if (path === this.currentRoute) return
    
    // Update browser history
    if (replace) {
      window.history.replaceState({}, '', path)
      // Replace current route in history
      if (this.navigationHistory.length > 0) {
        this.navigationHistory[this.navigationHistory.length - 1] = path
      } else {
        this.navigationHistory.push(path)
      }
    } else {
      window.history.pushState({}, '', path)
      // Add to navigation history
      this.navigationHistory.push(path)
      
      // Keep history limited to last 50 entries
      if (this.navigationHistory.length > 50) {
        this.navigationHistory = this.navigationHistory.slice(-50)
      }
    }
    
    this.handleRoute(path)
  }
  
  // Debug utilities
  getRegisteredRoutes(): string[] {
    return Array.from(this.routes.keys())
  }
  
  debugInfo(): any {
    return {
      currentRoute: this.currentRoute,
      registeredRoutes: this.getRegisteredRoutes(),
      isStarted: this.isStarted,
      queryParams: Object.fromEntries(this.getQueryParams()),
      hash: this.getHash(),
      navigationHistory: this.getNavigationHistory(),
      canGoBack: this.canGoBack(),
      previousRoute: this.getPreviousRoute(),
      securityContext: this.authGuard.getSecurityContext()
    }
  }
}
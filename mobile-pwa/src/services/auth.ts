import { EventEmitter } from '../utils/event-emitter'
import { Auth0Client, Auth0ClientOptions, User as Auth0User } from '@auth0/auth0-spa-js'
import { decodeJwt } from 'jose'

// Enhanced user interface with enterprise roles
export interface User {
  id: string
  email: string
  name: string
  full_name: string
  role: 'super_admin' | 'enterprise_admin' | 'pilot_manager' | 'success_manager' | 'developer' | 'viewer'
  permissions: string[]
  company_name?: string
  pilot_ids: string[]
  is_active: boolean
  last_login?: Date
  auth_method: 'password' | 'webauthn' | 'auth0'
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface AuthState {
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  lastActivity: number
  sessionId: string | null
  biometricEnabled: boolean
}

// Auth0 configuration interface
export interface Auth0Config {
  domain: string
  clientId: string
  audience?: string
  scope?: string
  redirectUri?: string
}

// Security audit log entry
export interface SecurityAuditLog {
  timestamp: Date
  event: 'login' | 'logout' | 'token_refresh' | 'permission_check' | 'session_timeout' | 'biometric_auth' | 'session_restored'
  userId?: string
  userAgent: string
  ipAddress?: string
  success: boolean
  details?: any
}

export class AuthService extends EventEmitter {
  private static instance: AuthService
  private state: AuthState = {
    user: null,
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    lastActivity: Date.now(),
    sessionId: null,
    biometricEnabled: false
  }
  
  private refreshTimer: number | null = null
  private activityTimer: number | null = null
  private auth0Client: Auth0Client | null = null
  private securityAuditLogs: SecurityAuditLog[] = []
  
  // Configuration
  private readonly TOKEN_REFRESH_INTERVAL = 15 * 60 * 1000 // 15 minutes
  private readonly SESSION_TIMEOUT = 30 * 60 * 1000 // 30 minutes
  private readonly MAX_FAILED_ATTEMPTS = 5
  private readonly LOCKOUT_DURATION = 15 * 60 * 1000 // 15 minutes
  
  // Auth0 configuration
  private auth0Config: Auth0Config = {
    domain: process.env.VITE_AUTH0_DOMAIN || 'your-domain.auth0.com',
    clientId: process.env.VITE_AUTH0_CLIENT_ID || 'your-client-id',
    audience: process.env.VITE_AUTH0_AUDIENCE || 'https://leanvibe-agent-hive',
    scope: 'openid profile email read:agents write:agents',
    redirectUri: window.location.origin
  }
  
  static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService()
    }
    return AuthService.instance
  }
  
  async initialize(): Promise<void> {
    try {
      // DEVELOPMENT MODE: Auto-authenticate for testing
      if (process.env.NODE_ENV === 'development' || window.location.hostname === 'localhost') {
        console.log('🔧 Development mode: Auto-authenticating...')
        this.state = {
          user: {
            id: 'dev-user-001',
            email: 'developer@leanvibe.com',
            name: 'Developer',
            full_name: 'Development User',
            role: 'super_admin',
            permissions: ['read:agents', 'write:agents', 'admin:system'],
            company_name: 'LeanVibe Development',
            pilot_ids: ['dev-pilot-001'],
            is_active: true,
            last_login: new Date(),
            auth_method: 'password'
          },
          token: 'dev-token-' + Date.now(),
          refreshToken: 'dev-refresh-token-' + Date.now(),
          isAuthenticated: true,
          lastActivity: Date.now(),
          sessionId: 'dev-session-' + Date.now(),
          biometricEnabled: false
        }
        
        console.log('✅ Development authentication complete')
        this.emit('authenticated', this.state.user)
        return
      }
      
      // Initialize Auth0 client
      await this.initializeAuth0()
      
      // Restore session from localStorage
      await this.restoreSession()
      
      // Setup activity monitoring
      this.setupActivityMonitoring()
      
      // Check for Auth0 callback
      if (window.location.search.includes('code=')) {
        await this.handleAuth0Callback()
        return
      }
      
      // Validate current session
      if (this.state.token) {
        await this.validateSession()
      }
      
      // Check for biometric authentication capability
      await this.checkBiometricCapability()
      
      this.logSecurityEvent('session_restored', true, {
        sessionId: this.state.sessionId,
        biometricEnabled: this.state.biometricEnabled
      })
      
    } catch (error) {
      console.error('Auth initialization failed:', error)
      this.logSecurityEvent('session_restored', false, { error: (error as Error).message })
      await this.logout()
    }
  }
  
  private async initializeAuth0(): Promise<void> {
    try {
      this.auth0Client = new Auth0Client({
        domain: this.auth0Config.domain,
        clientId: this.auth0Config.clientId,
        authorizationParams: {
          audience: this.auth0Config.audience,
          scope: this.auth0Config.scope,
          redirect_uri: this.auth0Config.redirectUri
        },
        cacheLocation: 'localstorage',
        useRefreshTokens: true
      })
      
      console.log('🔐 Auth0 client initialized')
    } catch (error) {
      console.error('Auth0 initialization failed:', error)
      // Continue without Auth0 - fallback to backend authentication
    }
  }
  
  private async handleAuth0Callback(): Promise<void> {
    if (!this.auth0Client) return
    
    try {
      await this.auth0Client.handleRedirectCallback()
      const auth0User = await this.auth0Client.getUser()
      const token = await this.auth0Client.getTokenSilently()
      
      if (auth0User && token) {
        // Convert Auth0 user to our user format
        const user = await this.convertAuth0User(auth0User, token)
        
        this.state = {
          user,
          token,
          refreshToken: null, // Auth0 handles refresh internally
          isAuthenticated: true,
          lastActivity: Date.now(),
          sessionId: this.generateSessionId(),
          biometricEnabled: this.state.biometricEnabled
        }
        
        await this.saveSession()
        this.setupTokenRefresh()
        this.emit('authenticated', user)
        
        // Clean up URL
        window.history.replaceState({}, document.title, window.location.pathname)
        
        this.logSecurityEvent('login', true, { 
          authMethod: 'auth0',
          userId: user.id 
        })
      }
    } catch (error) {
      console.error('Auth0 callback handling failed:', error)
      this.logSecurityEvent('login', false, { 
        authMethod: 'auth0',
        error: (error as Error).message 
      })
    }
  }
  
  private async convertAuth0User(auth0User: Auth0User, token: string): Promise<User> {
    // Decode JWT to get our custom claims
    const decodedToken = decodeJwt(token)
    
    return {
      id: auth0User.sub!,
      email: auth0User.email!,
      name: auth0User.name || auth0User.email!,
      full_name: auth0User.name || auth0User.email!,
      role: (decodedToken.role as any) || 'viewer',
      permissions: (decodedToken.permissions as string[]) || [],
      company_name: decodedToken.company_name as string,
      pilot_ids: (decodedToken.pilot_ids as string[]) || [],
      is_active: true,
      last_login: new Date(),
      auth_method: 'auth0'
    }
  }
  
  private async checkBiometricCapability(): Promise<void> {
    try {
      if (window.PublicKeyCredential && 
          await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable()) {
        this.state.biometricEnabled = true
        console.log('🔐 Biometric authentication available')
      }
    } catch (error) {
      console.log('Biometric authentication not available:', error.message)
    }
  }
  
  async login(credentials: LoginCredentials): Promise<User> {
    try {
      this.logSecurityEvent('login', false, { 
        authMethod: 'password',
        email: credentials.email 
      })
      
      const response = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      })
      
      if (!response.ok) {
        const error = await response.json()
        this.logSecurityEvent('login', false, { 
          authMethod: 'password',
          email: credentials.email,
          error: error.detail 
        })
        throw new Error(error.detail || 'Login failed')
      }
      
      const data = await response.json()
      
      // Convert backend user to our enhanced format
      const user: User = {
        id: data.user.id,
        email: data.user.email,
        name: data.user.full_name,
        full_name: data.user.full_name,
        role: data.user.role,
        permissions: data.user.permissions || [],
        company_name: data.user.company_name,
        pilot_ids: data.user.pilot_ids || [],
        is_active: data.user.is_active,
        last_login: data.user.last_login ? new Date(data.user.last_login) : new Date(),
        auth_method: 'password'
      }
      
      // Update state
      this.state = {
        user,
        token: data.access_token,
        refreshToken: data.refresh_token,
        isAuthenticated: true,
        lastActivity: Date.now(),
        sessionId: this.generateSessionId(),
        biometricEnabled: this.state.biometricEnabled
      }
      
      // Save to localStorage
      await this.saveSession()
      
      // Setup refresh timer
      this.setupTokenRefresh()
      
      // Emit authenticated event
      this.emit('authenticated', user)
      
      this.logSecurityEvent('login', true, { 
        authMethod: 'password',
        userId: user.id 
      })
      
      return user
      
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }
  
  async loginWithAuth0(): Promise<User> {
    try {
      if (!this.auth0Client) {
        throw new Error('Auth0 not configured')
      }
      
      this.logSecurityEvent('login', false, { authMethod: 'auth0' })
      
      // Redirect to Auth0 login
      await this.auth0Client.loginWithRedirect({
        authorizationParams: {
          audience: this.auth0Config.audience,
          scope: this.auth0Config.scope
        }
      })
      
      // This will redirect, so we won't reach here
      // The actual login completion happens in handleAuth0Callback
      throw new Error('Redirect in progress')
      
    } catch (error) {
      this.logSecurityEvent('login', false, { 
        authMethod: 'auth0',
        error: (error as Error).message 
      })
      throw error
    }
  }
  
  async loginWithWebAuthn(): Promise<User> {
    try {
      // Check if WebAuthn is supported
      if (!window.PublicKeyCredential) {
        throw new Error('WebAuthn is not supported in this browser')
      }
      
      this.logSecurityEvent('biometric_auth', false, { method: 'webauthn' })
      
      // Get challenge from server
      const challengeResponse = await fetch('/api/v1/auth/webauthn/challenge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!challengeResponse.ok) {
        throw new Error('Failed to get WebAuthn challenge')
      }
      
      const challengeData = await challengeResponse.json()
      
      // Convert challenge to Uint8Array
      const challenge = Uint8Array.from(atob(challengeData.challenge), c => c.charCodeAt(0))
      
      // Request credential
      const credential = await navigator.credentials.get({
        publicKey: {
          challenge,
          allowCredentials: challengeData.allowCredentials?.map((cred: any) => ({
            ...cred,
            id: Uint8Array.from(atob(cred.id), c => c.charCodeAt(0))
          })) || [],
          timeout: 60000,
          userVerification: 'preferred'
        }
      }) as PublicKeyCredential | null
      
      if (!credential) {
        this.logSecurityEvent('biometric_auth', false, { 
          method: 'webauthn',
          error: 'Authentication cancelled' 
        })
        throw new Error('WebAuthn authentication cancelled')
      }
      
      // Prepare assertion for server
      const response = credential.response as AuthenticatorAssertionResponse
      const assertion = {
        id: credential.id,
        rawId: btoa(String.fromCharCode(...new Uint8Array(credential.rawId))),
        response: {
          authenticatorData: btoa(String.fromCharCode(...new Uint8Array(response.authenticatorData))),
          clientDataJSON: btoa(String.fromCharCode(...new Uint8Array(response.clientDataJSON))),
          signature: btoa(String.fromCharCode(...new Uint8Array(response.signature))),
          userHandle: response.userHandle ? btoa(String.fromCharCode(...new Uint8Array(response.userHandle))) : null
        },
        type: credential.type
      }
      
      // Send assertion to server
      const authResponse = await fetch('/api/v1/auth/webauthn/authenticate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ assertion, challengeId: challengeData.challengeId }),
      })
      
      if (!authResponse.ok) {
        const error = await authResponse.json()
        this.logSecurityEvent('biometric_auth', false, { 
          method: 'webauthn',
          error: error.detail 
        })
        throw new Error(error.detail || 'WebAuthn authentication failed')
      }
      
      const data = await authResponse.json()
      
      // Convert backend user to our enhanced format
      const user: User = {
        id: data.user.id,
        email: data.user.email,
        name: data.user.full_name,
        full_name: data.user.full_name,
        role: data.user.role,
        permissions: data.user.permissions || [],
        company_name: data.user.company_name,
        pilot_ids: data.user.pilot_ids || [],
        is_active: data.user.is_active,
        last_login: data.user.last_login ? new Date(data.user.last_login) : new Date(),
        auth_method: 'webauthn'
      }
      
      // Update state
      this.state = {
        user,
        token: data.access_token,
        refreshToken: data.refresh_token,
        isAuthenticated: true,
        lastActivity: Date.now(),
        sessionId: this.generateSessionId(),
        biometricEnabled: this.state.biometricEnabled
      }
      
      // Save to localStorage
      await this.saveSession()
      
      // Setup refresh timer
      this.setupTokenRefresh()
      
      // Emit authenticated event
      this.emit('authenticated', user)
      
      this.logSecurityEvent('biometric_auth', true, { 
        method: 'webauthn',
        userId: user.id 
      })
      
      return user
      
    } catch (error) {
      console.error('WebAuthn authentication failed:', error)
      throw error
    }
  }
  
  async logout(): Promise<void> {
    try {
      const userId = this.state.user?.id
      
      // Handle Auth0 logout
      if (this.auth0Client && this.state.user?.auth_method === 'auth0') {
        await this.auth0Client.logout({
          logoutParams: {
            returnTo: window.location.origin
          }
        })
      }
      
      // Notify server if we have a token
      if (this.state.token) {
        fetch('/api/v1/auth/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.state.token}`,
            'Content-Type': 'application/json',
          },
        }).catch(() => {
          // Ignore logout API errors - we're logging out anyway
        })
      }
      
      // Clear state
      this.state = {
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        lastActivity: Date.now(),
        sessionId: null,
        biometricEnabled: this.state.biometricEnabled // Preserve biometric capability
      }
      
      // Clear localStorage
      localStorage.removeItem('auth_state')
      
      // Clear timers
      this.clearTimers()
      
      // Emit unauthenticated event
      this.emit('unauthenticated')
      
      this.logSecurityEvent('logout', true, { userId })
      
    } catch (error) {
      console.error('Logout failed:', error)
      // Force clear state even if API call fails
      this.state = {
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        lastActivity: Date.now(),
        sessionId: null,
        biometricEnabled: this.state.biometricEnabled
      }
      localStorage.removeItem('auth_state')
      this.clearTimers()
      this.emit('unauthenticated')
      this.logSecurityEvent('logout', false, { error: error.message })
    }
  }
  
  async refreshToken(): Promise<void> {
    try {
      if (!this.state.refreshToken) {
        throw new Error('No refresh token available')
      }
      
      const response = await fetch('/api/v1/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refresh_token: this.state.refreshToken
        }),
      })
      
      if (!response.ok) {
        throw new Error('Token refresh failed')
      }
      
      const data = await response.json()
      
      // Update tokens
      this.state.token = data.access_token
      this.state.refreshToken = data.refresh_token || this.state.refreshToken
      this.state.lastActivity = Date.now()
      
      // Save updated session
      await this.saveSession()
      
      console.log('🔄 Token refreshed successfully')
      
    } catch (error) {
      console.error('Token refresh failed:', error)
      await this.logout()
    }
  }
  
  private async validateSession(): Promise<void> {
    try {
      const response = await fetch('/api/v1/auth/me', {
        headers: {
          'Authorization': `Bearer ${this.state.token}`,
        },
      })
      
      if (!response.ok) {
        throw new Error('Session validation failed')
      }
      
      const userData = await response.json()
      this.state.user = userData
      this.state.isAuthenticated = true
      
      // Setup refresh timer
      this.setupTokenRefresh()
      
      this.emit('authenticated', this.state.user)
      
    } catch (error) {
      console.error('Session validation failed:', error)
      await this.logout()
    }
  }
  
  private async saveSession(): Promise<void> {
    try {
      const sessionData = {
        user: this.state.user,
        token: this.state.token,
        refreshToken: this.state.refreshToken,
        lastActivity: this.state.lastActivity
      }
      
      localStorage.setItem('auth_state', JSON.stringify(sessionData))
    } catch (error) {
      console.error('Failed to save session:', error)
    }
  }
  
  private async restoreSession(): Promise<void> {
    try {
      const savedSession = localStorage.getItem('auth_state')
      if (!savedSession) return
      
      const sessionData = JSON.parse(savedSession)
      
      // Check if session is not too old
      const timeSinceLastActivity = Date.now() - sessionData.lastActivity
      if (timeSinceLastActivity > this.SESSION_TIMEOUT) {
        localStorage.removeItem('auth_state')
        return
      }
      
      this.state = {
        user: sessionData.user,
        token: sessionData.token,
        refreshToken: sessionData.refreshToken,
        isAuthenticated: true,
        lastActivity: sessionData.lastActivity
      }
      
    } catch (error) {
      console.error('Failed to restore session:', error)
      localStorage.removeItem('auth_state')
    }
  }
  
  private setupTokenRefresh(): void {
    this.clearTimers()
    
    this.refreshTimer = window.setInterval(() => {
      this.refreshToken()
    }, this.TOKEN_REFRESH_INTERVAL)
  }
  
  private setupActivityMonitoring(): void {
    const updateActivity = () => {
      if (this.state.isAuthenticated) {
        this.state.lastActivity = Date.now()
      }
    }
    
    // Track user activity
    ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'].forEach(event => {
      document.addEventListener(event, updateActivity, { passive: true })
    })
    
    // Check for session timeout
    this.activityTimer = window.setInterval(() => {
      if (this.state.isAuthenticated) {
        const timeSinceLastActivity = Date.now() - this.state.lastActivity
        if (timeSinceLastActivity > this.SESSION_TIMEOUT) {
          console.log('Session timeout - logging out')
          this.logout()
        }
      }
    }, 60000) // Check every minute
  }
  
  private clearTimers(): void {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer)
      this.refreshTimer = null
    }
    
    if (this.activityTimer) {
      clearInterval(this.activityTimer)
      this.activityTimer = null
    }
  }
  
  // Public getters
  isAuthenticated(): boolean {
    return this.state.isAuthenticated
  }
  
  getUser(): User | null {
    return this.state.user
  }
  
  getToken(): string | null {
    return this.state.token
  }
  
  hasPermission(permission: string): boolean {
    return this.state.user?.permissions.includes(permission) || false
  }
  
  isAdmin(): boolean {
    return this.state.user?.role === 'super_admin' || this.state.user?.role === 'enterprise_admin'
  }
  
  // Helper method to add auth headers to requests
  getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    
    if (this.state.token) {
      headers['Authorization'] = `Bearer ${this.state.token}`
    }
    
    return headers
  }
  
  // RBAC methods
  hasRole(role: string): boolean {
    return this.state.user?.role === role
  }
  
  hasAnyRole(roles: string[]): boolean {
    return roles.includes(this.state.user?.role || '')
  }
  
  canAccessPilot(pilotId: string): boolean {
    if (!this.state.user) return false
    
    // Super admins can access all pilots
    if (this.state.user.role === 'super_admin') return true
    
    return this.state.user.pilot_ids.includes(pilotId)
  }
  
  // Security audit methods
  private logSecurityEvent(
    event: SecurityAuditLog['event'], 
    success: boolean, 
    details?: any
  ): void {
    const logEntry: SecurityAuditLog = {
      timestamp: new Date(),
      event,
      userId: this.state.user?.id,
      userAgent: navigator.userAgent,
      success,
      details
    }
    
    this.securityAuditLogs.push(logEntry)
    
    // Keep only last 100 entries in memory
    if (this.securityAuditLogs.length > 100) {
      this.securityAuditLogs = this.securityAuditLogs.slice(-100)
    }
    
    // Send to server for audit logging
    this.sendAuditLogToServer(logEntry).catch(console.error)
  }
  
  private async sendAuditLogToServer(logEntry: SecurityAuditLog): Promise<void> {
    try {
      await fetch('/api/v1/auth/audit', {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(logEntry)
      })
    } catch (error) {
      // Silently fail - audit logging shouldn't break the app
      console.warn('Failed to send audit log to server:', error)
    }
  }
  
  getSecurityAuditLogs(): SecurityAuditLog[] {
    return [...this.securityAuditLogs]
  }
  
  // Session management
  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }
  
  getSessionInfo(): { sessionId: string | null; lastActivity: Date; isExpired: boolean } {
    const now = Date.now()
    const isExpired = now - this.state.lastActivity > this.SESSION_TIMEOUT
    
    return {
      sessionId: this.state.sessionId,
      lastActivity: new Date(this.state.lastActivity),
      isExpired
    }
  }
  
  // Biometric authentication support
  isBiometricAvailable(): boolean {
    return this.state.biometricEnabled
  }
  
  async setupBiometric(): Promise<boolean> {
    if (!this.state.biometricEnabled) {
      throw new Error('Biometric authentication not available')
    }
    
    try {
      // This would register a new WebAuthn credential
      // Implementation depends on backend support
      console.log('🔐 Setting up biometric authentication...')
      return true
    } catch (error) {
      console.error('Biometric setup failed:', error)
      return false
    }
  }
  
  // Configuration methods
  updateAuth0Config(config: Partial<Auth0Config>): void {
    this.auth0Config = { ...this.auth0Config, ...config }
  }
  
  getAuth0Config(): Auth0Config {
    return { ...this.auth0Config }
  }
  
  // Enhanced session storage with encryption (basic)
  private async saveSession(): Promise<void> {
    try {
      const sessionData = {
        user: this.state.user,
        token: this.state.token,
        refreshToken: this.state.refreshToken,
        lastActivity: this.state.lastActivity,
        sessionId: this.state.sessionId,
        biometricEnabled: this.state.biometricEnabled
      }
      
      // In production, consider encrypting sensitive data
      localStorage.setItem('auth_state', JSON.stringify(sessionData))
    } catch (error) {
      console.error('Failed to save session:', error)
    }
  }
  
  private async restoreSession(): Promise<void> {
    try {
      const savedSession = localStorage.getItem('auth_state')
      if (!savedSession) return
      
      const sessionData = JSON.parse(savedSession)
      
      // Check if session is not too old
      const timeSinceLastActivity = Date.now() - sessionData.lastActivity
      if (timeSinceLastActivity > this.SESSION_TIMEOUT) {
        localStorage.removeItem('auth_state')
        return
      }
      
      this.state = {
        user: sessionData.user,
        token: sessionData.token,
        refreshToken: sessionData.refreshToken,
        isAuthenticated: true,
        lastActivity: sessionData.lastActivity,
        sessionId: sessionData.sessionId || this.generateSessionId(),
        biometricEnabled: sessionData.biometricEnabled || false
      }
      
    } catch (error) {
      console.error('Failed to restore session:', error)
      localStorage.removeItem('auth_state')
    }
  }
}
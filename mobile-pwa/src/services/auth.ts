import { EventEmitter } from '../utils/event-emitter'

export interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'observer'
  permissions: string[]
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
}

export class AuthService extends EventEmitter {
  private static instance: AuthService
  private state: AuthState = {
    user: null,
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    lastActivity: Date.now()
  }
  
  private refreshTimer: number | null = null
  private activityTimer: number | null = null
  private readonly TOKEN_REFRESH_INTERVAL = 15 * 60 * 1000 // 15 minutes
  private readonly SESSION_TIMEOUT = 30 * 60 * 1000 // 30 minutes
  
  static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService()
    }
    return AuthService.instance
  }
  
  async initialize(): Promise<void> {
    try {
      // Restore session from localStorage
      await this.restoreSession()
      
      // Setup activity monitoring
      this.setupActivityMonitoring()
      
      // Validate current session
      if (this.state.token) {
        await this.validateSession()
      }
      
    } catch (error) {
      console.error('Auth initialization failed:', error)
      await this.logout()
    }
  }
  
  async login(credentials: LoginCredentials): Promise<User> {
    try {
      const response = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Login failed')
      }
      
      const data = await response.json()
      
      // Update state
      this.state = {
        user: data.user,
        token: data.access_token,
        refreshToken: data.refresh_token,
        isAuthenticated: true,
        lastActivity: Date.now()
      }
      
      // Save to localStorage
      await this.saveSession()
      
      // Setup refresh timer
      this.setupTokenRefresh()
      
      // Emit authenticated event
      this.emit('authenticated', this.state.user)
      
      return this.state.user
      
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }
  
  async loginWithWebAuthn(): Promise<User> {
    try {
      // Check if WebAuthn is supported
      if (!window.PublicKeyCredential) {
        throw new Error('WebAuthn is not supported in this browser')
      }
      
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
        throw new Error(error.detail || 'WebAuthn authentication failed')
      }
      
      const data = await authResponse.json()
      
      // Update state
      this.state = {
        user: data.user,
        token: data.access_token,
        refreshToken: data.refresh_token,
        isAuthenticated: true,
        lastActivity: Date.now()
      }
      
      // Save to localStorage
      await this.saveSession()
      
      // Setup refresh timer
      this.setupTokenRefresh()
      
      // Emit authenticated event
      this.emit('authenticated', this.state.user)
      
      return this.state.user
      
    } catch (error) {
      console.error('WebAuthn authentication failed:', error)
      throw error
    }
  }
  
  async logout(): Promise<void> {
    try {
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
        lastActivity: Date.now()
      }
      
      // Clear localStorage
      localStorage.removeItem('auth_state')
      
      // Clear timers
      this.clearTimers()
      
      // Emit unauthenticated event
      this.emit('unauthenticated')
      
    } catch (error) {
      console.error('Logout failed:', error)
      // Force clear state even if API call fails
      this.state = {
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        lastActivity: Date.now()
      }
      localStorage.removeItem('auth_state')
      this.clearTimers()
      this.emit('unauthenticated')
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
    return this.state.user?.role === 'admin'
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
}
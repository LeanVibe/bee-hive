/**
 * Authentication Guards for Enterprise PWA
 * 
 * Provides comprehensive route protection, role-based access control,
 * and security validation for the LeanVibe Agent Hive dashboard.
 */

import { AuthService, User } from './auth'

export interface RouteGuardOptions {
  requireAuth?: boolean
  requiredRoles?: string[]
  requiredPermissions?: string[]
  requiredPilotAccess?: string[]
  allowGuest?: boolean
  redirectTo?: string
}

export interface GuardResult {
  allowed: boolean
  reason?: string
  redirectTo?: string
  user?: User
}

export class AuthGuard {
  private static instance: AuthGuard
  private authService: AuthService
  
  constructor() {
    this.authService = AuthService.getInstance()
  }
  
  static getInstance(): AuthGuard {
    if (!AuthGuard.instance) {
      AuthGuard.instance = new AuthGuard()
    }
    return AuthGuard.instance
  }
  
  /**
   * Main guard function for route protection
   */
  async canActivate(path: string, options: RouteGuardOptions = {}): Promise<GuardResult> {
    const {
      requireAuth = true,
      requiredRoles = [],
      requiredPermissions = [],
      requiredPilotAccess = [],
      allowGuest = false,
      redirectTo = '/login'
    } = options
    
    const user = this.authService.getUser()
    const isAuthenticated = this.authService.isAuthenticated()
    
    // Check if authentication is required
    if (requireAuth && !isAuthenticated) {
      if (allowGuest) {
        return {
          allowed: true,
          reason: 'Guest access allowed',
          user: undefined
        }
      }
      
      return {
        allowed: false,
        reason: 'Authentication required',
        redirectTo
      }
    }
    
    // If no authentication required and user is not authenticated
    if (!requireAuth && !isAuthenticated) {
      return {
        allowed: true,
        reason: 'No authentication required',
        user: undefined
      }
    }
    
    // User is authenticated - perform additional checks
    if (user) {
      // Check if user is active
      if (!user.is_active) {
        return {
          allowed: false,
          reason: 'User account is inactive',
          redirectTo: '/account-inactive'
        }
      }
      
      // Check role requirements
      if (requiredRoles.length > 0 && !this.hasRequiredRoles(user, requiredRoles)) {
        return {
          allowed: false,
          reason: `Insufficient role permissions. Required: ${requiredRoles.join(', ')}`,
          redirectTo: '/access-denied'
        }
      }
      
      // Check permission requirements
      if (requiredPermissions.length > 0 && !this.hasRequiredPermissions(user, requiredPermissions)) {
        return {
          allowed: false,
          reason: `Insufficient permissions. Required: ${requiredPermissions.join(', ')}`,
          redirectTo: '/access-denied'
        }
      }
      
      // Check pilot access requirements
      if (requiredPilotAccess.length > 0 && !this.hasRequiredPilotAccess(user, requiredPilotAccess)) {
        return {
          allowed: false,
          reason: `Insufficient pilot access. Required: ${requiredPilotAccess.join(', ')}`,
          redirectTo: '/access-denied'
        }
      }
      
      // Check session validity
      const sessionInfo = this.authService.getSessionInfo()
      if (sessionInfo.isExpired) {
        return {
          allowed: false,
          reason: 'Session expired',
          redirectTo: '/login'
        }
      }
    }
    
    return {
      allowed: true,
      reason: 'Access granted',
      user: user || undefined
    }
  }
  
  /**
   * Specific guards for common scenarios
   */
  
  async canAccessAdminRoutes(): Promise<GuardResult> {
    return this.canActivate('/admin', {
      requireAuth: true,
      requiredRoles: ['super_admin', 'enterprise_admin']
    })
  }
  
  async canAccessPilotManagement(): Promise<GuardResult> {
    return this.canActivate('/pilots', {
      requireAuth: true,
      requiredRoles: ['super_admin', 'enterprise_admin', 'pilot_manager']
    })
  }
  
  async canAccessDevelopmentTools(): Promise<GuardResult> {
    return this.canActivate('/development', {
      requireAuth: true,
      requiredRoles: ['super_admin', 'enterprise_admin', 'pilot_manager', 'developer'],
      requiredPermissions: ['execute_development_task']
    })
  }
  
  async canAccessROIMetrics(): Promise<GuardResult> {
    return this.canActivate('/roi', {
      requireAuth: true,
      requiredPermissions: ['view_roi_metrics']
    })
  }
  
  async canAccessExecutiveEngagement(): Promise<GuardResult> {
    return this.canActivate('/executive', {
      requireAuth: true,
      requiredRoles: ['super_admin', 'enterprise_admin', 'success_manager'],
      requiredPermissions: ['view_executive_engagement']
    })
  }
  
  async canAccessPilot(pilotId: string): Promise<GuardResult> {
    return this.canActivate(`/pilot/${pilotId}`, {
      requireAuth: true,
      requiredPilotAccess: [pilotId]
    })
  }
  
  /**
   * Helper methods for permission checking
   */
  
  private hasRequiredRoles(user: User, requiredRoles: string[]): boolean {
    return requiredRoles.includes(user.role)
  }
  
  private hasRequiredPermissions(user: User, requiredPermissions: string[]): boolean {
    return requiredPermissions.every(permission => 
      user.permissions.includes(permission)
    )
  }
  
  private hasRequiredPilotAccess(user: User, requiredPilotIds: string[]): boolean {
    // Super admins have access to all pilots
    if (user.role === 'super_admin') {
      return true
    }
    
    // Check if user has access to all required pilots
    return requiredPilotIds.every(pilotId => 
      user.pilot_ids.includes(pilotId)
    )
  }
  
  /**
   * Route protection decorator/helper
   */
  
  async protectRoute(
    routeHandler: () => void,
    path: string,
    options: RouteGuardOptions = {}
  ): Promise<void> {
    const guardResult = await this.canActivate(path, options)
    
    if (!guardResult.allowed) {
      if (guardResult.redirectTo) {
        // Emit navigation event
        window.dispatchEvent(new CustomEvent('auth-redirect', {
          detail: {
            path: guardResult.redirectTo,
            reason: guardResult.reason
          }
        }))
      } else {
        // Emit access denied event
        window.dispatchEvent(new CustomEvent('auth-access-denied', {
          detail: {
            path,
            reason: guardResult.reason
          }
        }))
      }
      return
    }
    
    // Route is allowed - execute handler
    routeHandler()
  }
  
  /**
   * Middleware for components
   */
  
  createComponentGuard(options: RouteGuardOptions) {
    return async (component: any) => {
      const guardResult = await this.canActivate(window.location.pathname, options)
      
      if (!guardResult.allowed) {
        // Return access denied component or redirect
        return {
          allowed: false,
          reason: guardResult.reason,
          redirectTo: guardResult.redirectTo
        }
      }
      
      return {
        allowed: true,
        component,
        user: guardResult.user
      }
    }
  }
  
  /**
   * Batch permission checking for UI elements
   */
  
  async checkMultiplePermissions(checks: Array<{
    name: string
    options: RouteGuardOptions
  }>): Promise<Record<string, boolean>> {
    const results: Record<string, boolean> = {}
    
    for (const check of checks) {
      const result = await this.canActivate('', check.options)
      results[check.name] = result.allowed
    }
    
    return results
  }
  
  /**
   * Dynamic permission checking for conditional UI rendering
   */
  
  canUserPerformAction(action: string, context?: any): boolean {
    const user = this.authService.getUser()
    if (!user) return false
    
    switch (action) {
      case 'create_agent':
        return this.hasRequiredPermissions(user, ['create_development_task'])
      
      case 'delete_agent':
        return this.hasRequiredRoles(user, ['super_admin', 'enterprise_admin'])
      
      case 'view_audit_logs':
        return this.hasRequiredRoles(user, ['super_admin', 'enterprise_admin'])
      
      case 'manage_users':
        return this.hasRequiredPermissions(user, ['manage_users'])
      
      case 'access_pilot':
        if (context?.pilotId) {
          return this.hasRequiredPilotAccess(user, [context.pilotId])
        }
        return false
      
      case 'view_system_logs':
        return this.hasRequiredPermissions(user, ['view_system_logs'])
      
      case 'configure_system':
        return this.hasRequiredPermissions(user, ['configure_system'])
      
      default:
        return false
    }
  }
  
  /**
   * Security context for components
   */
  
  getSecurityContext(): {
    user: User | null
    isAuthenticated: boolean
    sessionInfo: any
    permissions: string[]
    canPerform: (action: string, context?: any) => boolean
  } {
    const user = this.authService.getUser()
    const sessionInfo = this.authService.getSessionInfo()
    
    return {
      user,
      isAuthenticated: this.authService.isAuthenticated(),
      sessionInfo,
      permissions: user?.permissions || [],
      canPerform: (action: string, context?: any) => this.canUserPerformAction(action, context)
    }
  }
}

// Singleton instance
export const authGuard = AuthGuard.getInstance()
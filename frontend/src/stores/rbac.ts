/**
 * RBAC Store - Enterprise Role-Based Access Control State Management
 * 
 * Manages roles, permissions, user assignments, and permission matrix
 * for Epic 6 Phase 2 Advanced User Management System.
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'

// Types and Interfaces
export interface Role {
  id: string
  name: string
  description?: string
  permissions: Permission[]
  is_system_role: boolean
  is_active: boolean
  user_count: number
  created_at: string
  updated_at: string
  parent_role?: string
}

export interface Permission {
  value: string
  name: string
  category: string
  description: string
}

export interface PermissionMatrixEntry {
  role_name: string
  permission: string
  granted: boolean
  inherited: boolean
  source_role?: string
}

export interface PermissionMatrix {
  roles: string[]
  permissions: string[]
  matrix: PermissionMatrixEntry[]
  last_updated: string
}

export interface UserRoleAssignment {
  user_id: string
  role_ids: string[]
  assigned_by?: string
  expires_at?: string
}

export interface BulkRoleAssignment {
  user_ids: string[]
  role_ids: string[]
  assigned_by?: string
  expires_at?: string
}

export interface RoleHierarchy {
  role_id: string
  role_name: string
  parent_id?: string
  children: RoleHierarchy[]
  depth: number
  user_count: number
}

// Permission Categories for UI Organization
export const PERMISSION_CATEGORIES = {
  'Pilot Management': [
    'create_pilot', 'view_pilot', 'update_pilot', 'delete_pilot'
  ],
  'Analytics & ROI': [
    'view_roi_metrics', 'create_roi_metrics'
  ],
  'Executive Engagement': [
    'view_executive_engagement', 'create_executive_engagement', 'update_executive_engagement'
  ],
  'Development': [
    'create_development_task', 'view_development_task', 'execute_development_task'
  ],
  'System Administration': [
    'manage_users', 'view_system_logs', 'configure_system'
  ]
} as const

export const useRBACStore = defineStore('rbac', () => {
  // State
  const roles = ref<Role[]>([])
  const permissions = ref<Permission[]>([])
  const permissionMatrix = ref<PermissionMatrix | null>(null)
  const roleHierarchy = ref<RoleHierarchy[]>([])
  const userRoles = ref<Map<string, Role[]>>(new Map())
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // Search and filter state
  const searchQuery = ref('')
  const selectedCategory = ref<string | null>(null)
  const showInactiveRoles = ref(false)

  // Getters
  const activeRoles = computed(() => 
    roles.value.filter(role => role.is_active || showInactiveRoles.value)
  )

  const filteredRoles = computed(() => {
    let filtered = activeRoles.value

    if (searchQuery.value) {
      const query = searchQuery.value.toLowerCase()
      filtered = filtered.filter(role => 
        role.name.toLowerCase().includes(query) ||
        (role.description?.toLowerCase().includes(query))
      )
    }

    return filtered.sort((a, b) => {
      // System roles first, then alphabetical
      if (a.is_system_role && !b.is_system_role) return -1
      if (!a.is_system_role && b.is_system_role) return 1
      return a.name.localeCompare(b.name)
    })
  })

  const systemRoles = computed(() => 
    roles.value.filter(role => role.is_system_role && role.is_active)
  )

  const customRoles = computed(() => 
    roles.value.filter(role => !role.is_system_role && role.is_active)
  )

  const permissionsByCategory = computed(() => {
    const grouped: Record<string, Permission[]> = {}
    
    permissions.value.forEach(permission => {
      if (!grouped[permission.category]) {
        grouped[permission.category] = []
      }
      grouped[permission.category].push(permission)
    })

    return grouped
  })

  const roleStats = computed(() => ({
    total: roles.value.length,
    active: roles.value.filter(r => r.is_active).length,
    system: systemRoles.value.length,
    custom: customRoles.value.length,
    totalUsers: roles.value.reduce((sum, role) => sum + role.user_count, 0)
  }))

  // Actions
  const fetchRoles = async (options: {
    includeInactive?: boolean
    search?: string
  } = {}) => {
    isLoading.value = true
    error.value = null

    try {
      const params = new URLSearchParams()
      if (options.includeInactive) params.set('include_inactive', 'true')
      if (options.search) params.set('search', options.search)

      const response = await apiService.get(`/api/rbac/roles?${params}`)
      roles.value = response.data
      
      console.log(`Fetched ${response.data.length} roles`)
    } catch (err: any) {
      error.value = err.message || 'Failed to fetch roles'
      console.error('Error fetching roles:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const createRole = async (roleData: {
    name: string
    description?: string
    permissions: string[]
    is_system_role?: boolean
    parent_role?: string
  }): Promise<Role> => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.post('/api/rbac/roles', roleData)
      const newRole = response.data
      
      roles.value.push(newRole)
      console.log(`Created role: ${newRole.name}`)
      
      return newRole
    } catch (err: any) {
      error.value = err.message || 'Failed to create role'
      console.error('Error creating role:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const updateRole = async (
    roleId: string, 
    updates: {
      name?: string
      description?: string
      permissions?: string[]
      is_active?: boolean
    }
  ): Promise<Role> => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.put(`/api/rbac/roles/${roleId}`, updates)
      const updatedRole = response.data
      
      const index = roles.value.findIndex(r => r.id === roleId)
      if (index !== -1) {
        roles.value[index] = updatedRole
      }
      
      console.log(`Updated role: ${updatedRole.name}`)
      return updatedRole
    } catch (err: any) {
      error.value = err.message || 'Failed to update role'
      console.error('Error updating role:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const deleteRole = async (roleId: string): Promise<void> => {
    isLoading.value = true
    error.value = null

    try {
      await apiService.delete(`/api/rbac/roles/${roleId}`)
      
      const index = roles.value.findIndex(r => r.id === roleId)
      if (index !== -1) {
        roles.value.splice(index, 1)
      }
      
      console.log(`Deleted role: ${roleId}`)
    } catch (err: any) {
      error.value = err.message || 'Failed to delete role'
      console.error('Error deleting role:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const fetchPermissions = async () => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.get('/api/rbac/permissions')
      permissions.value = response.data
      
      console.log(`Fetched ${response.data.length} permissions`)
    } catch (err: any) {
      error.value = err.message || 'Failed to fetch permissions'
      console.error('Error fetching permissions:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const fetchPermissionMatrix = async () => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.get('/api/rbac/permission-matrix')
      permissionMatrix.value = response.data
      
      console.log('Fetched permission matrix')
    } catch (err: any) {
      error.value = err.message || 'Failed to fetch permission matrix'
      console.error('Error fetching permission matrix:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const assignUserRoles = async (assignment: UserRoleAssignment): Promise<void> => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.post('/api/rbac/assign-roles', assignment)
      
      // Update local cache of user roles
      const assignedRoles = roles.value.filter(r => assignment.role_ids.includes(r.id))
      userRoles.value.set(assignment.user_id, assignedRoles)
      
      console.log(`Assigned ${assignment.role_ids.length} roles to user ${assignment.user_id}`)
      return response.data
    } catch (err: any) {
      error.value = err.message || 'Failed to assign roles'
      console.error('Error assigning roles:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const bulkAssignRoles = async (assignment: BulkRoleAssignment): Promise<void> => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.post('/api/rbac/bulk-assign-roles', assignment)
      
      // Update local cache for affected users
      const assignedRoles = roles.value.filter(r => assignment.role_ids.includes(r.id))
      assignment.user_ids.forEach(userId => {
        userRoles.value.set(userId, assignedRoles)
      })
      
      console.log(
        `Bulk assigned ${assignment.role_ids.length} roles to ${assignment.user_ids.length} users`
      )
      return response.data
    } catch (err: any) {
      error.value = err.message || 'Failed to bulk assign roles'
      console.error('Error bulk assigning roles:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const fetchUserRoles = async (userId: string): Promise<Role[]> => {
    try {
      const response = await apiService.get(`/api/rbac/user-roles/${userId}`)
      const userRoleData = response.data.roles
      
      userRoles.value.set(userId, userRoleData)
      return userRoleData
    } catch (err: any) {
      error.value = err.message || 'Failed to fetch user roles'
      console.error('Error fetching user roles:', err)
      throw err
    }
  }

  const fetchRoleHierarchy = async () => {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.get('/api/rbac/hierarchy')
      roleHierarchy.value = response.data
      
      console.log('Fetched role hierarchy')
    } catch (err: any) {
      error.value = err.message || 'Failed to fetch role hierarchy'
      console.error('Error fetching role hierarchy:', err)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  // Utility functions
  const getRoleById = (roleId: string): Role | undefined => {
    return roles.value.find(role => role.id === roleId)
  }

  const getRoleByName = (roleName: string): Role | undefined => {
    return roles.value.find(role => role.name === roleName)
  }

  const getUserRoles = (userId: string): Role[] => {
    return userRoles.value.get(userId) || []
  }

  const hasPermission = (userId: string, permission: string): boolean => {
    const userRoleList = getUserRoles(userId)
    return userRoleList.some(role => 
      role.permissions.some(perm => 
        typeof perm === 'string' ? perm === permission : perm.value === permission
      )
    )
  }

  const canManageRole = (role: Role, currentUserRoles: Role[]): boolean => {
    // Super admins can manage all roles
    if (currentUserRoles.some(r => r.name === 'Super Admin')) {
      return true
    }
    
    // System roles can only be managed by super admins
    if (role.is_system_role) {
      return false
    }
    
    // Regular admins can manage custom roles
    return currentUserRoles.some(r => r.name === 'Enterprise Admin')
  }

  // Filter and search functions
  const setSearchQuery = (query: string) => {
    searchQuery.value = query
  }

  const setSelectedCategory = (category: string | null) => {
    selectedCategory.value = category
  }

  const toggleShowInactive = () => {
    showInactiveRoles.value = !showInactiveRoles.value
  }

  const resetFilters = () => {
    searchQuery.value = ''
    selectedCategory.value = null
    showInactiveRoles.value = false
  }

  // Initialize store
  const initialize = async () => {
    try {
      await Promise.all([
        fetchRoles(),
        fetchPermissions(),
        fetchPermissionMatrix()
      ])
    } catch (err) {
      console.error('Error initializing RBAC store:', err)
    }
  }

  return {
    // State
    roles,
    permissions,
    permissionMatrix,
    roleHierarchy,
    userRoles,
    isLoading,
    error,
    searchQuery,
    selectedCategory,
    showInactiveRoles,

    // Getters
    activeRoles,
    filteredRoles,
    systemRoles,
    customRoles,
    permissionsByCategory,
    roleStats,

    // Actions
    fetchRoles,
    createRole,
    updateRole,
    deleteRole,
    fetchPermissions,
    fetchPermissionMatrix,
    assignUserRoles,
    bulkAssignRoles,
    fetchUserRoles,
    fetchRoleHierarchy,

    // Utility functions
    getRoleById,
    getRoleByName,
    getUserRoles,
    hasPermission,
    canManageRole,

    // Filter functions
    setSearchQuery,
    setSelectedCategory,
    toggleShowInactive,
    resetFilters,

    // Initialize
    initialize
  }
})
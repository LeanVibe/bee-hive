<template>
  <div class="min-h-screen bg-gray-50 p-6">
    <!-- Header -->
    <div class="mb-8">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-900">Role & Permission Management</h1>
          <p class="mt-1 text-gray-600">
            Manage user roles, permissions, and access control for your organization
          </p>
        </div>
        
        <div class="flex items-center space-x-4">
          <!-- Quick Stats -->
          <div class="flex items-center space-x-6 bg-white px-6 py-3 rounded-lg shadow-sm border">
            <div class="text-center">
              <div class="text-2xl font-bold text-blue-600">{{ rbacStore.roleStats.total }}</div>
              <div class="text-xs text-gray-500 uppercase tracking-wide">Total Roles</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-green-600">{{ rbacStore.roleStats.active }}</div>
              <div class="text-xs text-gray-500 uppercase tracking-wide">Active</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-purple-600">{{ rbacStore.roleStats.totalUsers }}</div>
              <div class="text-xs text-gray-500 uppercase tracking-wide">Users</div>
            </div>
          </div>
          
          <button
            @click="showCreateModal = true"
            class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            <span>New Role</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Main Content Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left Column: Role Management -->
      <div class="lg:col-span-2 space-y-6">
        <!-- Search and Filters -->
        <div class="bg-white rounded-lg shadow-sm border p-6">
          <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0 sm:space-x-4">
            <!-- Search -->
            <div class="flex-1 relative">
              <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                v-model="searchQuery"
                type="text"
                placeholder="Search roles..."
                class="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                @input="handleSearch"
              >
            </div>
            
            <!-- Filters -->
            <div class="flex items-center space-x-4">
              <label class="flex items-center text-sm">
                <input
                  v-model="showInactiveRoles"
                  type="checkbox"
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  @change="handleFilterChange"
                >
                <span class="ml-2 text-gray-700">Show inactive</span>
              </label>
              
              <button
                @click="refreshData"
                class="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100 transition-colors"
                :disabled="rbacStore.isLoading"
              >
                <svg 
                  class="w-5 h-5" 
                  :class="{ 'animate-spin': rbacStore.isLoading }" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        <!-- Roles Grid -->
        <div class="bg-white rounded-lg shadow-sm border">
          <div class="p-6 border-b border-gray-200">
            <div class="flex items-center justify-between">
              <h2 class="text-lg font-semibold text-gray-900">Roles</h2>
              <span class="text-sm text-gray-500">
                {{ rbacStore.filteredRoles.length }} of {{ rbacStore.roleStats.total }}
              </span>
            </div>
          </div>

          <div v-if="rbacStore.isLoading" class="p-12 text-center">
            <div class="inline-flex items-center space-x-2">
              <svg class="animate-spin h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span class="text-gray-600">Loading roles...</span>
            </div>
          </div>

          <div v-else-if="rbacStore.error" class="p-12 text-center text-red-600">
            <svg class="mx-auto h-12 w-12 text-red-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <p>{{ rbacStore.error }}</p>
            <button 
              @click="refreshData"
              class="mt-4 text-blue-600 hover:text-blue-500 font-medium"
            >
              Try again
            </button>
          </div>

          <div v-else class="divide-y divide-gray-200">
            <RoleCard
              v-for="role in rbacStore.filteredRoles"
              :key="role.id"
              :role="role"
              @edit="handleEditRole"
              @delete="handleDeleteRole"
              @view-users="handleViewRoleUsers"
            />
          </div>

          <div v-if="!rbacStore.isLoading && !rbacStore.error && rbacStore.filteredRoles.length === 0" 
               class="p-12 text-center text-gray-500">
            <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
            </svg>
            <p class="text-lg font-medium mb-2">No roles found</p>
            <p class="mb-4">{{ searchQuery ? 'No roles match your search criteria.' : 'Get started by creating your first custom role.' }}</p>
            <button
              v-if="!searchQuery"
              @click="showCreateModal = true"
              class="inline-flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              <span>Create Role</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Right Column: Quick Actions & Info -->
      <div class="space-y-6">
        <!-- Permission Matrix Preview -->
        <div class="bg-white rounded-lg shadow-sm border">
          <div class="p-6 border-b border-gray-200">
            <div class="flex items-center justify-between">
              <h3 class="text-lg font-semibold text-gray-900">Permission Matrix</h3>
              <button
                @click="showMatrixModal = true"
                class="text-sm text-blue-600 hover:text-blue-500 font-medium"
              >
                View Full Matrix
              </button>
            </div>
          </div>
          
          <div class="p-6">
            <div v-if="rbacStore.permissionMatrix" class="space-y-3">
              <div class="text-sm text-gray-600 mb-4">
                {{ rbacStore.permissionMatrix.roles.length }} roles × {{ rbacStore.permissionMatrix.permissions.length }} permissions
              </div>
              
              <!-- Mini matrix preview -->
              <div class="grid grid-cols-4 gap-1 text-xs">
                <div 
                  v-for="(role, index) in rbacStore.permissionMatrix.roles.slice(0, 4)"
                  :key="role"
                  class="p-2 bg-gray-50 rounded text-center truncate"
                  :title="role"
                >
                  {{ role }}
                </div>
              </div>
            </div>
            
            <div v-else class="text-center text-gray-500 py-4">
              <svg class="mx-auto h-8 w-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <p class="text-sm">Load matrix to view permissions</p>
            </div>
          </div>
        </div>

        <!-- Quick Actions -->
        <div class="bg-white rounded-lg shadow-sm border">
          <div class="p-6 border-b border-gray-200">
            <h3 class="text-lg font-semibold text-gray-900">Quick Actions</h3>
          </div>
          
          <div class="p-6 space-y-4">
            <button
              @click="showBulkAssignModal = true"
              class="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg hover:from-blue-100 hover:to-indigo-100 transition-colors text-blue-700"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              <span>Bulk Assign Roles</span>
            </button>
            
            <button
              @click="exportRoles"
              class="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg hover:from-green-100 hover:to-emerald-100 transition-colors text-green-700"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>Export Roles</span>
            </button>
            
            <button
              @click="showAuditLog = true"
              class="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gradient-to-r from-purple-50 to-violet-50 border border-purple-200 rounded-lg hover:from-purple-100 hover:to-violet-100 transition-colors text-purple-700"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>View Audit Log</span>
            </button>
          </div>
        </div>

        <!-- Recent Activity -->
        <div class="bg-white rounded-lg shadow-sm border">
          <div class="p-6 border-b border-gray-200">
            <h3 class="text-lg font-semibold text-gray-900">Recent Activity</h3>
          </div>
          
          <div class="p-6">
            <div class="space-y-3">
              <!-- Activity items would come from audit log -->
              <div v-for="i in 3" :key="i" class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                <div class="text-sm">
                  <p class="text-gray-900">Role assignment updated</p>
                  <p class="text-gray-500">{{ formatTimeAgo(new Date(Date.now() - i * 3600000)) }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Modals -->
    <RoleCreationModal 
      v-if="showCreateModal" 
      @close="showCreateModal = false"
      @created="handleRoleCreated"
    />
    
    <PermissionMatrix 
      v-if="showMatrixModal" 
      @close="showMatrixModal = false"
    />
    
    <BulkRoleAssignmentModal 
      v-if="showBulkAssignModal" 
      @close="showBulkAssignModal = false"
    />

    <!-- Toast Notifications -->
    <div class="fixed bottom-4 right-4 z-50">
      <div
        v-if="notification"
        class="bg-white border border-gray-200 rounded-lg shadow-lg p-4 mb-2 max-w-sm transform transition-all duration-300"
        :class="notification.type === 'success' ? 'border-l-4 border-l-green-500' : 'border-l-4 border-l-red-500'"
      >
        <div class="flex items-center space-x-3">
          <svg 
            v-if="notification.type === 'success'"
            class="w-5 h-5 text-green-500" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          <svg 
            v-else
            class="w-5 h-5 text-red-500" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
          <p class="text-sm font-medium">{{ notification.message }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRBACStore, type Role } from '@/stores/rbac'
import RoleCard from './RoleCard.vue'
import RoleCreationModal from './RoleCreationModal.vue'
import PermissionMatrix from './PermissionMatrix.vue'
import BulkRoleAssignmentModal from './BulkRoleAssignmentModal.vue'

const rbacStore = useRBACStore()

// Modal states
const showCreateModal = ref(false)
const showMatrixModal = ref(false)
const showBulkAssignModal = ref(false)
const showAuditLog = ref(false)

// Search and filter states
const searchQuery = ref('')
const showInactiveRoles = ref(false)

// Notification state
const notification = ref<{
  type: 'success' | 'error'
  message: string
} | null>(null)

// Computed properties
const formatTimeAgo = (date: Date): string => {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  
  if (diffHours < 1) return 'Just now'
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  return `${diffDays}d ago`
}

// Event handlers
const handleSearch = () => {
  rbacStore.setSearchQuery(searchQuery.value)
}

const handleFilterChange = () => {
  rbacStore.toggleShowInactive()
}

const refreshData = async () => {
  try {
    await Promise.all([
      rbacStore.fetchRoles({ includeInactive: showInactiveRoles.value }),
      rbacStore.fetchPermissionMatrix()
    ])
    
    showNotification('success', 'Data refreshed successfully')
  } catch (error) {
    showNotification('error', 'Failed to refresh data')
  }
}

const handleEditRole = (role: Role) => {
  console.log('Edit role:', role.name)
  // TODO: Implement role editing modal
}

const handleDeleteRole = async (role: Role) => {
  if (role.is_system_role) {
    showNotification('error', 'Cannot delete system roles')
    return
  }

  if (confirm(`Are you sure you want to delete role "${role.name}"?`)) {
    try {
      await rbacStore.deleteRole(role.id)
      showNotification('success', `Role "${role.name}" deleted successfully`)
    } catch (error) {
      showNotification('error', 'Failed to delete role')
    }
  }
}

const handleViewRoleUsers = (role: Role) => {
  console.log('View users for role:', role.name)
  // TODO: Implement role users modal
}

const handleRoleCreated = (role: Role) => {
  showNotification('success', `Role "${role.name}" created successfully`)
  showCreateModal.value = false
}

const exportRoles = () => {
  const rolesData = rbacStore.filteredRoles.map(role => ({
    name: role.name,
    description: role.description,
    permissions: role.permissions,
    user_count: role.user_count,
    is_system_role: role.is_system_role,
    is_active: role.is_active
  }))

  const blob = new Blob([JSON.stringify(rolesData, null, 2)], { 
    type: 'application/json' 
  })
  
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `roles_export_${new Date().toISOString().split('T')[0]}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)

  showNotification('success', 'Roles exported successfully')
}

const showNotification = (type: 'success' | 'error', message: string) => {
  notification.value = { type, message }
  setTimeout(() => {
    notification.value = null
  }, 5000)
}

// Watch for search query changes
watch(searchQuery, (newValue) => {
  rbacStore.setSearchQuery(newValue)
})

watch(showInactiveRoles, (newValue) => {
  rbacStore.showInactiveRoles = newValue
})

// Initialize on mount
onMounted(async () => {
  await rbacStore.initialize()
})
</script>

<style scoped>
/* Custom styles for enhanced UI */
.role-card-hover {
  @apply transform hover:scale-[1.01] transition-all duration-200;
}
</style>
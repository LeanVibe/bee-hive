<template>
  <div class="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
    <!-- Backdrop -->
    <div class="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
      <div 
        class="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" 
        @click="$emit('close')"
        aria-hidden="true"
      ></div>

      <!-- Modal panel -->
      <div class="relative inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-7xl sm:w-full">
        <!-- Header -->
        <div class="bg-white px-6 py-4 border-b border-gray-200">
          <div class="flex items-center justify-between">
            <div>
              <h3 class="text-xl font-semibold text-gray-900" id="modal-title">
                Permission Matrix
              </h3>
              <p class="mt-1 text-sm text-gray-600">
                View and manage permissions across all roles in your organization
              </p>
            </div>
            <div class="flex items-center space-x-4">
              <!-- Export Button -->
              <button
                @click="exportMatrix"
                class="inline-flex items-center space-x-2 px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>Export</span>
              </button>
              
              <!-- Close Button -->
              <button
                @click="$emit('close')"
                class="text-gray-400 hover:text-gray-600 focus:outline-none focus:text-gray-600 transition-colors"
              >
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        <!-- Content -->
        <div class="bg-white px-6 py-6 max-h-[80vh] overflow-auto">
          <!-- Loading State -->
          <div v-if="rbacStore.isLoading" class="flex items-center justify-center h-64">
            <div class="text-center">
              <svg class="animate-spin h-8 w-8 text-blue-600 mx-auto mb-4" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p class="text-gray-600">Loading permission matrix...</p>
            </div>
          </div>

          <!-- Error State -->
          <div v-else-if="rbacStore.error" class="text-center py-12">
            <svg class="mx-auto h-12 w-12 text-red-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <p class="text-red-600 mb-4">{{ rbacStore.error }}</p>
            <button 
              @click="loadMatrix"
              class="text-blue-600 hover:text-blue-500 font-medium"
            >
              Try again
            </button>
          </div>

          <!-- Matrix Content -->
          <div v-else-if="matrixData" class="space-y-6">
            <!-- Matrix Stats -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div class="bg-blue-50 rounded-lg p-4 text-center">
                <div class="text-2xl font-bold text-blue-600">{{ matrixData.roles.length }}</div>
                <div class="text-sm text-blue-600">Active Roles</div>
              </div>
              <div class="bg-green-50 rounded-lg p-4 text-center">
                <div class="text-2xl font-bold text-green-600">{{ matrixData.permissions.length }}</div>
                <div class="text-sm text-green-600">Total Permissions</div>
              </div>
              <div class="bg-purple-50 rounded-lg p-4 text-center">
                <div class="text-2xl font-bold text-purple-600">{{ totalGrantedPermissions }}</div>
                <div class="text-sm text-purple-600">Granted Permissions</div>
              </div>
            </div>

            <!-- Filters and Search -->
            <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0 sm:space-x-4 mb-6">
              <!-- Search -->
              <div class="flex-1 relative max-w-md">
                <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
                <input
                  v-model="searchQuery"
                  type="text"
                  placeholder="Search roles or permissions..."
                  class="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                >
              </div>

              <!-- Category Filter -->
              <select
                v-model="selectedCategory"
                class="px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                <option value="">All Categories</option>
                <option
                  v-for="category in permissionCategories"
                  :key="category"
                  :value="category"
                >
                  {{ category }}
                </option>
              </select>

              <!-- View Toggle -->
              <div class="flex items-center space-x-2">
                <button
                  @click="viewMode = 'grid'"
                  class="p-2 rounded-md"
                  :class="viewMode === 'grid' ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:text-gray-600'"
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                  </svg>
                </button>
                <button
                  @click="viewMode = 'compact'"
                  class="p-2 rounded-md"
                  :class="viewMode === 'compact' ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:text-gray-600'"
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                  </svg>
                </button>
              </div>
            </div>

            <!-- Permission Matrix Table -->
            <div class="border border-gray-200 rounded-lg overflow-hidden">
              <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                  <!-- Table Header -->
                  <thead class="bg-gray-50">
                    <tr>
                      <th class="sticky left-0 z-10 bg-gray-50 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200">
                        Role / Permission
                      </th>
                      <th
                        v-for="permission in filteredPermissions"
                        :key="permission.value"
                        class="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider min-w-[120px] border-r border-gray-200"
                        :title="permission.description"
                      >
                        <div class="transform -rotate-45 whitespace-nowrap origin-bottom-left">
                          {{ formatPermissionName(permission.name) }}
                        </div>
                      </th>
                    </tr>
                  </thead>

                  <!-- Table Body -->
                  <tbody class="bg-white divide-y divide-gray-200">
                    <tr
                      v-for="role in filteredRoles"
                      :key="role"
                      class="hover:bg-gray-50"
                    >
                      <!-- Role Name (Sticky Column) -->
                      <td class="sticky left-0 z-10 bg-white px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 border-r border-gray-200 hover:bg-gray-50">
                        <div class="flex items-center space-x-2">
                          <!-- Role Type Indicator -->
                          <div 
                            class="w-3 h-3 rounded-full"
                            :class="isSystemRole(role) ? 'bg-blue-500' : 'bg-green-500'"
                            :title="isSystemRole(role) ? 'System Role' : 'Custom Role'"
                          ></div>
                          <span>{{ role }}</span>
                        </div>
                      </td>

                      <!-- Permission Cells -->
                      <td
                        v-for="permission in filteredPermissions"
                        :key="`${role}-${permission.value}`"
                        class="px-3 py-4 text-center border-r border-gray-200"
                      >
                        <div class="flex justify-center">
                          <div
                            class="w-6 h-6 rounded-full flex items-center justify-center transition-all duration-200"
                            :class="getPermissionStatus(role, permission.value)"
                            :title="getPermissionTooltip(role, permission.value)"
                          >
                            <svg
                              v-if="hasPermission(role, permission.value)"
                              class="w-4 h-4 text-white"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </svg>
                            <svg
                              v-else
                              class="w-4 h-4 text-gray-300"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </div>
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Matrix Summary -->
            <div class="bg-gray-50 rounded-lg p-4">
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center">
                  <div class="text-lg font-semibold text-gray-900">{{ filteredRoles.length }}</div>
                  <div class="text-sm text-gray-600">Roles Shown</div>
                </div>
                <div class="text-center">
                  <div class="text-lg font-semibold text-gray-900">{{ filteredPermissions.length }}</div>
                  <div class="text-sm text-gray-600">Permissions Shown</div>
                </div>
                <div class="text-center">
                  <div class="text-lg font-semibold text-gray-900">{{ filteredGrantedPermissions }}</div>
                  <div class="text-sm text-gray-600">Granted (Shown)</div>
                </div>
                <div class="text-center">
                  <div class="text-lg font-semibold text-gray-900">{{ Math.round(grantedPercentage) }}%</div>
                  <div class="text-sm text-gray-600">Grant Rate</div>
                </div>
              </div>
            </div>

            <!-- Legend -->
            <div class="bg-white border border-gray-200 rounded-lg p-4">
              <h4 class="text-sm font-medium text-gray-900 mb-3">Legend</h4>
              <div class="flex flex-wrap items-center space-x-6">
                <div class="flex items-center space-x-2">
                  <div class="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center">
                    <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <span class="text-sm text-gray-700">Permission Granted</span>
                </div>
                <div class="flex items-center space-x-2">
                  <div class="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center">
                    <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </div>
                  <span class="text-sm text-gray-700">Permission Denied</span>
                </div>
                <div class="flex items-center space-x-2">
                  <div class="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span class="text-sm text-gray-700">System Role</span>
                </div>
                <div class="flex items-center space-x-2">
                  <div class="w-3 h-3 rounded-full bg-green-500"></div>
                  <span class="text-sm text-gray-700">Custom Role</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="bg-gray-50 px-6 py-3 border-t border-gray-200">
          <div class="flex items-center justify-between">
            <div class="text-sm text-gray-500">
              Last updated: {{ matrixData ? formatDate(matrixData.last_updated) : 'Never' }}
            </div>
            <button
              @click="loadMatrix"
              class="inline-flex items-center space-x-2 px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              :disabled="rbacStore.isLoading"
            >
              <svg 
                class="w-4 h-4" 
                :class="{ 'animate-spin': rbacStore.isLoading }" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>{{ rbacStore.isLoading ? 'Refreshing...' : 'Refresh' }}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRBACStore, PERMISSION_CATEGORIES, type PermissionMatrix as PermissionMatrixType } from '@/stores/rbac'

interface Emits {
  (e: 'close'): void
}

const emit = defineEmits<Emits>()
const rbacStore = useRBACStore()

// Component state
const searchQuery = ref('')
const selectedCategory = ref('')
const viewMode = ref<'grid' | 'compact'>('grid')

// Computed properties
const matrixData = computed(() => rbacStore.permissionMatrix)

const permissionCategories = computed(() => {
  return Object.keys(PERMISSION_CATEGORIES)
})

const filteredRoles = computed(() => {
  if (!matrixData.value) return []
  
  let roles = matrixData.value.roles
  
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    roles = roles.filter(role => role.toLowerCase().includes(query))
  }
  
  return roles
})

const filteredPermissions = computed(() => {
  if (!matrixData.value) return []
  
  let permissions = rbacStore.permissions
  
  if (selectedCategory.value) {
    const categoryPermissions = PERMISSION_CATEGORIES[selectedCategory.value as keyof typeof PERMISSION_CATEGORIES]
    permissions = permissions.filter(p => categoryPermissions.includes(p.value as any))
  }
  
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    permissions = permissions.filter(p => 
      p.name.toLowerCase().includes(query) || 
      p.description.toLowerCase().includes(query)
    )
  }
  
  return permissions
})

const totalGrantedPermissions = computed(() => {
  if (!matrixData.value) return 0
  return matrixData.value.matrix.filter(entry => entry.granted).length
})

const filteredGrantedPermissions = computed(() => {
  if (!matrixData.value) return 0
  
  let count = 0
  for (const role of filteredRoles.value) {
    for (const permission of filteredPermissions.value) {
      if (hasPermission(role, permission.value)) {
        count++
      }
    }
  }
  return count
})

const grantedPercentage = computed(() => {
  const total = filteredRoles.value.length * filteredPermissions.value.length
  return total > 0 ? (filteredGrantedPermissions.value / total) * 100 : 0
})

// Helper methods
const formatPermissionName = (name: string): string => {
  return name.replace(/([A-Z])/g, ' $1').trim()
}

const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleString()
}

const isSystemRole = (roleName: string): boolean => {
  const role = rbacStore.roles.find(r => r.name === roleName)
  return role?.is_system_role || false
}

const hasPermission = (roleName: string, permissionValue: string): boolean => {
  if (!matrixData.value) return false
  
  const entry = matrixData.value.matrix.find(
    e => e.role_name === roleName && e.permission === permissionValue
  )
  
  return entry?.granted || false
}

const getPermissionStatus = (roleName: string, permissionValue: string): string => {
  const granted = hasPermission(roleName, permissionValue)
  
  if (granted) {
    return 'bg-green-500 hover:bg-green-600'
  } else {
    return 'bg-gray-200 hover:bg-gray-300'
  }
}

const getPermissionTooltip = (roleName: string, permissionValue: string): string => {
  const granted = hasPermission(roleName, permissionValue)
  const permission = rbacStore.permissions.find(p => p.value === permissionValue)
  
  return `${roleName} ${granted ? 'has' : 'does not have'} ${permission?.name || permissionValue} permission`
}

// Actions
const loadMatrix = async () => {
  await rbacStore.fetchPermissionMatrix()
}

const exportMatrix = () => {
  if (!matrixData.value) return
  
  // Create CSV format
  const headers = ['Role', ...filteredPermissions.value.map(p => p.name)]
  const rows = [headers]
  
  filteredRoles.value.forEach(role => {
    const row = [role]
    filteredPermissions.value.forEach(permission => {
      row.push(hasPermission(role, permission.value) ? 'Yes' : 'No')
    })
    rows.push(row)
  })
  
  const csvContent = rows.map(row => row.join(',')).join('\n')
  const blob = new Blob([csvContent], { type: 'text/csv' })
  
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `permission_matrix_${new Date().toISOString().split('T')[0]}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// Initialize on mount
onMounted(async () => {
  if (!matrixData.value) {
    await loadMatrix()
  }
})
</script>

<style scoped>
/* Custom scrollbar for table */
.overflow-x-auto::-webkit-scrollbar {
  height: 8px;
}

.overflow-x-auto::-webkit-scrollbar-track {
  background: #f1f5f9;
}

.overflow-x-auto::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}

.overflow-x-auto::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* Sticky column shadow */
.sticky.left-0 {
  box-shadow: 2px 0 4px -2px rgba(0, 0, 0, 0.1);
}

/* Rotate headers */
.transform.-rotate-45 {
  transform: rotate(-45deg);
  transform-origin: bottom left;
  width: 120px;
  height: 60px;
  display: flex;
  align-items: end;
  justify-content: start;
  padding-left: 10px;
  padding-bottom: 5px;
}
</style>
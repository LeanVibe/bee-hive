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
      <div class="relative inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full sm:p-6">
        <!-- Header -->
        <div class="flex items-center justify-between mb-6">
          <div>
            <h3 class="text-lg leading-6 font-semibold text-gray-900" id="modal-title">
              Bulk Role Assignment
            </h3>
            <p class="mt-1 text-sm text-gray-600">
              Assign roles to multiple users at once. Select users and roles to apply in batch.
            </p>
          </div>
          <button
            @click="$emit('close')"
            class="text-gray-400 hover:text-gray-600 focus:outline-none focus:text-gray-600 transition-colors"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <!-- Progress Steps -->
        <div class="mb-8">
          <nav aria-label="Progress">
            <ol class="flex items-center justify-center">
              <li
                v-for="(step, index) in steps"
                :key="step.id"
                :class="[
                  index !== steps.length - 1 ? 'pr-8 sm:pr-20' : '',
                  'relative'
                ]"
              >
                <div v-if="index !== steps.length - 1" class="absolute inset-0 flex items-center" aria-hidden="true">
                  <div class="h-0.5 w-full bg-gray-200"></div>
                </div>
                <div
                  class="relative flex items-center justify-center w-8 h-8 rounded-full"
                  :class="[
                    currentStep >= index + 1
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border-2 border-gray-300 text-gray-500'
                  ]"
                >
                  <svg
                    v-if="currentStep > index + 1"
                    class="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                  </svg>
                  <span v-else class="text-sm font-medium">{{ index + 1 }}</span>
                </div>
                <div class="mt-2 text-xs text-center text-gray-500">{{ step.name }}</div>
              </li>
            </ol>
          </nav>
        </div>

        <!-- Step Content -->
        <div class="min-h-[400px]">
          <!-- Step 1: Select Users -->
          <div v-if="currentStep === 1" class="space-y-6">
            <div class="border-b border-gray-200 pb-4">
              <h4 class="text-lg font-medium text-gray-900">Select Users</h4>
              <p class="text-sm text-gray-600 mt-1">
                Choose the users you want to assign roles to. You can search by name or email.
              </p>
            </div>

            <!-- User Search -->
            <div class="relative">
              <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                v-model="userSearchQuery"
                type="text"
                placeholder="Search users by name or email..."
                class="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
              >
            </div>

            <!-- User Selection Options -->
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-4">
                <button
                  @click="selectAllUsers"
                  class="text-sm text-blue-600 hover:text-blue-500 font-medium"
                >
                  Select All ({{ filteredUsers.length }})
                </button>
                <button
                  @click="clearUserSelection"
                  class="text-sm text-gray-600 hover:text-gray-500 font-medium"
                >
                  Clear Selection
                </button>
              </div>
              <div class="text-sm text-gray-600">
                {{ selectedUsers.length }} of {{ filteredUsers.length }} selected
              </div>
            </div>

            <!-- Users List -->
            <div class="max-h-96 overflow-y-auto border border-gray-200 rounded-lg divide-y divide-gray-200">
              <div
                v-for="user in filteredUsers"
                :key="user.id"
                class="p-4 hover:bg-gray-50 cursor-pointer transition-colors"
                @click="toggleUserSelection(user)"
              >
                <div class="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    :checked="selectedUsers.includes(user.id)"
                    class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    @click.stop="toggleUserSelection(user)"
                  >
                  <div class="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center">
                    <span class="text-sm font-medium text-gray-600">
                      {{ getUserInitials(user.full_name) }}
                    </span>
                  </div>
                  <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium text-gray-900 truncate">{{ user.full_name }}</p>
                    <p class="text-sm text-gray-500 truncate">{{ user.email }}</p>
                    <div class="flex items-center space-x-2 mt-1">
                      <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                        {{ user.role }}
                      </span>
                      <span v-if="user.company_name" class="text-xs text-gray-500">
                        {{ user.company_name }}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="filteredUsers.length === 0" class="p-8 text-center text-gray-500">
                <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
                <p>No users found matching your search.</p>
              </div>
            </div>
          </div>

          <!-- Step 2: Select Roles -->
          <div v-if="currentStep === 2" class="space-y-6">
            <div class="border-b border-gray-200 pb-4">
              <h4 class="text-lg font-medium text-gray-900">Select Roles</h4>
              <p class="text-sm text-gray-600 mt-1">
                Choose the roles to assign to the selected {{ selectedUsers.length }} users.
              </p>
            </div>

            <!-- Role Selection Options -->
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-4">
                <button
                  @click="selectAllRoles"
                  class="text-sm text-blue-600 hover:text-blue-500 font-medium"
                >
                  Select All Roles
                </button>
                <button
                  @click="clearRoleSelection"
                  class="text-sm text-gray-600 hover:text-gray-500 font-medium"
                >
                  Clear Selection
                </button>
              </div>
              <div class="text-sm text-gray-600">
                {{ selectedRoles.length }} roles selected
              </div>
            </div>

            <!-- Roles Grid -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div
                v-for="role in availableRoles"
                :key="role.id"
                class="border border-gray-200 rounded-lg p-4 hover:border-blue-300 hover:bg-blue-50 cursor-pointer transition-all"
                :class="{
                  'border-blue-500 bg-blue-50': selectedRoles.includes(role.id),
                  'opacity-75': !role.is_active
                }"
                @click="toggleRoleSelection(role)"
              >
                <div class="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    :checked="selectedRoles.includes(role.id)"
                    class="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    @click.stop="toggleRoleSelection(role)"
                  >
                  <div class="flex-1">
                    <div class="flex items-center space-x-2">
                      <h5 class="text-sm font-medium text-gray-900">{{ role.name }}</h5>
                      <span
                        v-if="role.is_system_role"
                        class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                      >
                        System
                      </span>
                      <span
                        v-if="!role.is_active"
                        class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                      >
                        Inactive
                      </span>
                    </div>
                    <p v-if="role.description" class="text-xs text-gray-600 mt-1">
                      {{ role.description }}
                    </p>
                    <div class="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                      <span>{{ role.user_count }} users</span>
                      <span>{{ Array.isArray(role.permissions) ? role.permissions.length : 0 }} permissions</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Step 3: Review & Confirm -->
          <div v-if="currentStep === 3" class="space-y-6">
            <div class="border-b border-gray-200 pb-4">
              <h4 class="text-lg font-medium text-gray-900">Review Assignment</h4>
              <p class="text-sm text-gray-600 mt-1">
                Review the bulk role assignment before applying changes.
              </p>
            </div>

            <!-- Assignment Summary -->
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-6">
              <div class="flex items-center space-x-3 mb-4">
                <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h5 class="text-lg font-medium text-blue-900">Assignment Summary</h5>
              </div>
              
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div>
                  <h6 class="text-sm font-medium text-blue-900 mb-2">Selected Users ({{ selectedUsers.length }})</h6>
                  <div class="space-y-2 max-h-32 overflow-y-auto">
                    <div
                      v-for="userId in selectedUsers"
                      :key="userId"
                      class="text-sm text-blue-800"
                    >
                      {{ getUserById(userId)?.full_name }}
                    </div>
                  </div>
                </div>
                
                <div>
                  <h6 class="text-sm font-medium text-blue-900 mb-2">Selected Roles ({{ selectedRoles.length }})</h6>
                  <div class="space-y-2 max-h-32 overflow-y-auto">
                    <div
                      v-for="roleId in selectedRoles"
                      :key="roleId"
                      class="text-sm text-blue-800"
                    >
                      {{ getRoleById(roleId)?.name }}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Warning Messages -->
            <div v-if="hasSystemRoles" class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div class="flex">
                <svg class="w-5 h-5 text-yellow-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <div class="ml-3">
                  <h3 class="text-sm font-medium text-yellow-800">System Roles Warning</h3>
                  <div class="mt-1 text-sm text-yellow-700">
                    You are assigning system roles which have elevated privileges. Please ensure users need these permissions.
                  </div>
                </div>
              </div>
            </div>

            <!-- Impact Analysis -->
            <div class="bg-gray-50 rounded-lg p-4">
              <h6 class="text-sm font-medium text-gray-900 mb-3">Impact Analysis</h6>
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
                <div>
                  <div class="text-2xl font-bold text-gray-900">{{ selectedUsers.length * selectedRoles.length }}</div>
                  <div class="text-sm text-gray-600">Total Assignments</div>
                </div>
                <div>
                  <div class="text-2xl font-bold text-gray-900">{{ uniquePermissions.length }}</div>
                  <div class="text-sm text-gray-600">Unique Permissions</div>
                </div>
                <div>
                  <div class="text-2xl font-bold text-gray-900">{{ selectedUsers.length }}</div>
                  <div class="text-sm text-gray-600">Affected Users</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Navigation -->
        <div class="flex items-center justify-between pt-6 border-t border-gray-200">
          <button
            v-if="currentStep > 1"
            @click="previousStep"
            class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            :disabled="isSubmitting"
          >
            Previous
          </button>
          <div v-else></div>

          <div class="flex space-x-4">
            <button
              @click="$emit('close')"
              class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              :disabled="isSubmitting"
            >
              Cancel
            </button>

            <button
              v-if="currentStep < 3"
              @click="nextStep"
              :disabled="!canProceed"
              class="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>

            <button
              v-else
              @click="submitBulkAssignment"
              :disabled="!canProceed || isSubmitting"
              class="px-4 py-2 text-sm font-medium text-white bg-green-600 border border-transparent rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span v-if="isSubmitting" class="flex items-center space-x-2">
                <svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Assigning...</span>
              </span>
              <span v-else>Apply Assignment</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRBACStore, type Role } from '@/stores/rbac'

interface User {
  id: string
  email: string
  full_name: string
  role: string
  company_name?: string
}

interface Emits {
  (e: 'close'): void
}

const emit = defineEmits<Emits>()
const rbacStore = useRBACStore()

// Component state
const currentStep = ref(1)
const isSubmitting = ref(false)
const userSearchQuery = ref('')

// Form data
const selectedUsers = ref<string[]>([])
const selectedRoles = ref<string[]>([])

// Mock users data (in real app, this would come from API)
const allUsers = ref<User[]>([
  { id: '1', email: 'john@example.com', full_name: 'John Smith', role: 'Developer', company_name: 'TechCorp' },
  { id: '2', email: 'jane@example.com', full_name: 'Jane Doe', role: 'Manager', company_name: 'TechCorp' },
  { id: '3', email: 'bob@example.com', full_name: 'Bob Johnson', role: 'Analyst', company_name: 'DataCorp' },
  { id: '4', email: 'alice@example.com', full_name: 'Alice Brown', role: 'Designer', company_name: 'DesignCorp' },
  { id: '5', email: 'charlie@example.com', full_name: 'Charlie Wilson', role: 'Developer', company_name: 'TechCorp' }
])

// Steps configuration
const steps = [
  { id: 1, name: 'Select Users' },
  { id: 2, name: 'Select Roles' },
  { id: 3, name: 'Review & Confirm' }
]

// Computed properties
const filteredUsers = computed(() => {
  if (!userSearchQuery.value) return allUsers.value
  
  const query = userSearchQuery.value.toLowerCase()
  return allUsers.value.filter(user =>
    user.full_name.toLowerCase().includes(query) ||
    user.email.toLowerCase().includes(query) ||
    user.company_name?.toLowerCase().includes(query)
  )
})

const availableRoles = computed(() => {
  return rbacStore.roles.filter(role => role.is_active)
})

const hasSystemRoles = computed(() => {
  return selectedRoles.value.some(roleId => {
    const role = rbacStore.roles.find(r => r.id === roleId)
    return role?.is_system_role
  })
})

const uniquePermissions = computed(() => {
  const permissions = new Set<string>()
  selectedRoles.value.forEach(roleId => {
    const role = rbacStore.roles.find(r => r.id === roleId)
    if (role && Array.isArray(role.permissions)) {
      role.permissions.forEach(permission => {
        const permissionValue = typeof permission === 'string' ? permission : permission.value
        permissions.add(permissionValue)
      })
    }
  })
  return Array.from(permissions)
})

const canProceed = computed(() => {
  switch (currentStep.value) {
    case 1:
      return selectedUsers.value.length > 0
    case 2:
      return selectedRoles.value.length > 0
    case 3:
      return selectedUsers.value.length > 0 && selectedRoles.value.length > 0
    default:
      return false
  }
})

// Helper methods
const getUserInitials = (name: string): string => {
  return name
    .split(' ')
    .map(word => word.charAt(0).toUpperCase())
    .join('')
    .substring(0, 2)
}

const getUserById = (userId: string): User | undefined => {
  return allUsers.value.find(user => user.id === userId)
}

const getRoleById = (roleId: string): Role | undefined => {
  return rbacStore.roles.find(role => role.id === roleId)
}

// User selection methods
const toggleUserSelection = (user: User) => {
  const index = selectedUsers.value.indexOf(user.id)
  if (index > -1) {
    selectedUsers.value.splice(index, 1)
  } else {
    selectedUsers.value.push(user.id)
  }
}

const selectAllUsers = () => {
  selectedUsers.value = filteredUsers.value.map(user => user.id)
}

const clearUserSelection = () => {
  selectedUsers.value = []
}

// Role selection methods
const toggleRoleSelection = (role: Role) => {
  const index = selectedRoles.value.indexOf(role.id)
  if (index > -1) {
    selectedRoles.value.splice(index, 1)
  } else {
    selectedRoles.value.push(role.id)
  }
}

const selectAllRoles = () => {
  selectedRoles.value = availableRoles.value.map(role => role.id)
}

const clearRoleSelection = () => {
  selectedRoles.value = []
}

// Navigation methods
const nextStep = () => {
  if (canProceed.value && currentStep.value < 3) {
    currentStep.value++
  }
}

const previousStep = () => {
  if (currentStep.value > 1) {
    currentStep.value--
  }
}

// Submit bulk assignment
const submitBulkAssignment = async () => {
  if (!canProceed.value) return

  isSubmitting.value = true

  try {
    const bulkAssignment = {
      user_ids: selectedUsers.value,
      role_ids: selectedRoles.value,
      assigned_by: 'current_user', // TODO: Get from auth context
      expires_at: undefined // Optional expiration
    }

    await rbacStore.bulkAssignRoles(bulkAssignment)
    
    // Show success notification and close modal
    emit('close')
    
    // TODO: Show success toast notification
    console.log(`Successfully assigned ${selectedRoles.value.length} roles to ${selectedUsers.value.length} users`)
    
  } catch (error: any) {
    console.error('Error in bulk role assignment:', error)
    // TODO: Show error notification
  } finally {
    isSubmitting.value = false
  }
}

// Initialize component
onMounted(async () => {
  // Ensure roles are loaded
  if (rbacStore.roles.length === 0) {
    await rbacStore.fetchRoles()
  }
  
  // TODO: Load users from API
  // await loadUsers()
})
</script>

<style scoped>
/* Custom scrollbar */
.overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}

.overflow-y-auto::-webkit-scrollbar-track {
  background: #f1f5f9;
}

.overflow-y-auto::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

.overflow-y-auto::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* Step transition animations */
.step-enter-active,
.step-leave-active {
  transition: all 0.3s ease;
}

.step-enter-from {
  opacity: 0;
  transform: translateX(20px);
}

.step-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}
</style>
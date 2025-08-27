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
      <div class="relative inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-2xl sm:w-full sm:p-6">
        <!-- Header -->
        <div class="flex items-center justify-between mb-6">
          <div>
            <h3 class="text-lg leading-6 font-semibold text-gray-900" id="modal-title">
              Create New Role
            </h3>
            <p class="mt-1 text-sm text-gray-600">
              Define a custom role with specific permissions for your organization.
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

        <!-- Form -->
        <form @submit.prevent="createRole" class="space-y-6">
          <!-- Role Basic Info -->
          <div class="grid grid-cols-1 gap-6">
            <!-- Role Name -->
            <div>
              <label for="role-name" class="block text-sm font-medium text-gray-700 mb-2">
                Role Name *
              </label>
              <input
                id="role-name"
                v-model="formData.name"
                type="text"
                required
                class="block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                :class="{ 'border-red-300': errors.name }"
                placeholder="Enter role name (e.g., Project Manager)"
              >
              <p v-if="errors.name" class="mt-1 text-sm text-red-600">{{ errors.name }}</p>
            </div>

            <!-- Role Description -->
            <div>
              <label for="role-description" class="block text-sm font-medium text-gray-700 mb-2">
                Description
              </label>
              <textarea
                id="role-description"
                v-model="formData.description"
                rows="3"
                class="block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm resize-none"
                placeholder="Describe the role's responsibilities and scope..."
              ></textarea>
            </div>
          </div>

          <!-- Permissions Selection -->
          <div>
            <div class="flex items-center justify-between mb-4">
              <label class="block text-sm font-medium text-gray-700">
                Permissions *
              </label>
              <div class="flex items-center space-x-2">
                <button
                  type="button"
                  @click="selectAllPermissions"
                  class="text-sm text-blue-600 hover:text-blue-500 font-medium"
                >
                  Select All
                </button>
                <span class="text-gray-300">|</span>
                <button
                  type="button"
                  @click="clearAllPermissions"
                  class="text-sm text-gray-600 hover:text-gray-500 font-medium"
                >
                  Clear All
                </button>
              </div>
            </div>

            <!-- Permission Categories -->
            <div class="space-y-6">
              <div 
                v-for="(permissions, category) in rbacStore.permissionsByCategory" 
                :key="category"
                class="border border-gray-200 rounded-lg p-4"
              >
                <!-- Category Header -->
                <div class="flex items-center justify-between mb-3">
                  <div class="flex items-center space-x-2">
                    <input
                      :id="`category-${category}`"
                      type="checkbox"
                      :checked="isCategorySelected(category)"
                      :indeterminate="isCategoryIndeterminate(category)"
                      @change="toggleCategory(category)"
                      class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    >
                    <label :for="`category-${category}`" class="font-medium text-gray-900">
                      {{ category }}
                    </label>
                  </div>
                  <span class="text-sm text-gray-500">
                    {{ getCategorySelectionCount(category) }}/{{ permissions.length }}
                  </span>
                </div>

                <!-- Category Permissions -->
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 ml-6">
                  <label
                    v-for="permission in permissions"
                    :key="permission.value"
                    class="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                  >
                    <input
                      type="checkbox"
                      :value="permission.value"
                      v-model="formData.permissions"
                      class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    >
                    <div class="flex-1 min-w-0">
                      <div class="text-sm font-medium text-gray-900">{{ permission.name }}</div>
                      <div class="text-xs text-gray-500 truncate">{{ permission.description }}</div>
                    </div>
                  </label>
                </div>
              </div>
            </div>

            <p v-if="errors.permissions" class="mt-1 text-sm text-red-600">{{ errors.permissions }}</p>
          </div>

          <!-- Selected Permissions Summary -->
          <div v-if="formData.permissions.length > 0" class="bg-blue-50 rounded-lg p-4">
            <div class="flex items-center justify-between mb-2">
              <h4 class="text-sm font-medium text-blue-900">Selected Permissions</h4>
              <span class="text-sm text-blue-700">{{ formData.permissions.length }} selected</span>
            </div>
            <div class="flex flex-wrap gap-1">
              <span
                v-for="permissionValue in formData.permissions.slice(0, 10)"
                :key="permissionValue"
                class="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-800"
              >
                {{ formatPermissionName(permissionValue) }}
              </span>
              <span
                v-if="formData.permissions.length > 10"
                class="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-200 text-blue-700"
              >
                +{{ formData.permissions.length - 10 }} more
              </span>
            </div>
          </div>

          <!-- Advanced Options (Optional) -->
          <div class="border border-gray-200 rounded-lg p-4">
            <button
              type="button"
              @click="showAdvancedOptions = !showAdvancedOptions"
              class="flex items-center justify-between w-full text-left"
            >
              <span class="text-sm font-medium text-gray-700">Advanced Options</span>
              <svg 
                class="w-4 h-4 text-gray-500 transform transition-transform"
                :class="{ 'rotate-180': showAdvancedOptions }"
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            <div v-if="showAdvancedOptions" class="mt-4 space-y-4">
              <!-- Parent Role -->
              <div>
                <label for="parent-role" class="block text-sm font-medium text-gray-700 mb-2">
                  Parent Role (Inheritance)
                </label>
                <select
                  id="parent-role"
                  v-model="formData.parent_role"
                  class="block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                >
                  <option value="">No parent role</option>
                  <option
                    v-for="role in rbacStore.activeRoles.filter(r => r.id !== formData.name)"
                    :key="role.id"
                    :value="role.id"
                  >
                    {{ role.name }}
                  </option>
                </select>
                <p class="mt-1 text-xs text-gray-500">
                  Inherit permissions from parent role (optional)
                </p>
              </div>

              <!-- System Role Flag -->
              <div class="flex items-center">
                <input
                  id="system-role"
                  v-model="formData.is_system_role"
                  type="checkbox"
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                >
                <label for="system-role" class="ml-2 block text-sm text-gray-700">
                  System Role
                </label>
                <div class="ml-2">
                  <div class="group relative">
                    <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div class="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 text-xs text-white bg-gray-900 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
                      System roles cannot be deleted by regular admins
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Error Messages -->
          <div v-if="Object.keys(errors).length > 0" class="bg-red-50 border border-red-200 rounded-lg p-3">
            <div class="flex">
              <svg class="w-5 h-5 text-red-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <div class="ml-3">
                <h3 class="text-sm font-medium text-red-800">Please fix the following errors:</h3>
                <ul class="mt-1 text-sm text-red-700">
                  <li v-for="(error, field) in errors" :key="field">{{ error }}</li>
                </ul>
              </div>
            </div>
          </div>

          <!-- Actions -->
          <div class="flex items-center justify-end space-x-4 pt-4 border-t border-gray-200">
            <button
              type="button"
              @click="$emit('close')"
              class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              :disabled="isSubmitting"
            >
              Cancel
            </button>
            <button
              type="submit"
              class="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="isSubmitting || !isFormValid"
            >
              <span v-if="isSubmitting" class="flex items-center space-x-2">
                <svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Creating...</span>
              </span>
              <span v-else>Create Role</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRBACStore, type Role } from '@/stores/rbac'

interface Emits {
  (e: 'close'): void
  (e: 'created', role: Role): void
}

const emit = defineEmits<Emits>()
const rbacStore = useRBACStore()

// Form data
const formData = reactive({
  name: '',
  description: '',
  permissions: [] as string[],
  is_system_role: false,
  parent_role: ''
})

// Component state
const isSubmitting = ref(false)
const showAdvancedOptions = ref(false)
const errors = reactive<Record<string, string>>({})

// Computed properties
const isFormValid = computed(() => {
  return formData.name.trim().length > 0 && 
         formData.permissions.length > 0 &&
         Object.keys(errors).length === 0
})

// Permission category methods
const isCategorySelected = (category: string): boolean => {
  const categoryPermissions = rbacStore.permissionsByCategory[category]
  if (!categoryPermissions) return false
  
  return categoryPermissions.every(permission => 
    formData.permissions.includes(permission.value)
  )
}

const isCategoryIndeterminate = (category: string): boolean => {
  const categoryPermissions = rbacStore.permissionsByCategory[category]
  if (!categoryPermissions) return false
  
  const selectedCount = categoryPermissions.filter(permission =>
    formData.permissions.includes(permission.value)
  ).length
  
  return selectedCount > 0 && selectedCount < categoryPermissions.length
}

const getCategorySelectionCount = (category: string): number => {
  const categoryPermissions = rbacStore.permissionsByCategory[category]
  if (!categoryPermissions) return 0
  
  return categoryPermissions.filter(permission =>
    formData.permissions.includes(permission.value)
  ).length
}

const toggleCategory = (category: string) => {
  const categoryPermissions = rbacStore.permissionsByCategory[category]
  if (!categoryPermissions) return
  
  const isSelected = isCategorySelected(category)
  
  if (isSelected) {
    // Remove all permissions from this category
    formData.permissions = formData.permissions.filter(permissionValue =>
      !categoryPermissions.some(p => p.value === permissionValue)
    )
  } else {
    // Add all permissions from this category
    const categoryPermissionValues = categoryPermissions.map(p => p.value)
    const newPermissions = categoryPermissionValues.filter(value =>
      !formData.permissions.includes(value)
    )
    formData.permissions.push(...newPermissions)
  }
}

// Permission selection methods
const selectAllPermissions = () => {
  formData.permissions = rbacStore.permissions.map(p => p.value)
}

const clearAllPermissions = () => {
  formData.permissions = []
}

// Utility methods
const formatPermissionName = (permissionValue: string): string => {
  return permissionValue
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

// Validation
const validateForm = (): boolean => {
  const newErrors: Record<string, string> = {}

  // Validate name
  if (!formData.name.trim()) {
    newErrors.name = 'Role name is required'
  } else if (formData.name.length < 2) {
    newErrors.name = 'Role name must be at least 2 characters'
  } else if (formData.name.length > 100) {
    newErrors.name = 'Role name must be less than 100 characters'
  } else {
    // Check for duplicate names
    const existingRole = rbacStore.roles.find(role => 
      role.name.toLowerCase() === formData.name.toLowerCase()
    )
    if (existingRole) {
      newErrors.name = 'A role with this name already exists'
    }
  }

  // Validate permissions
  if (formData.permissions.length === 0) {
    newErrors.permissions = 'At least one permission must be selected'
  }

  // Clear previous errors and set new ones
  Object.keys(errors).forEach(key => delete errors[key])
  Object.assign(errors, newErrors)

  return Object.keys(newErrors).length === 0
}

// Form submission
const createRole = async () => {
  if (!validateForm()) return

  isSubmitting.value = true

  try {
    const roleData = {
      name: formData.name.trim(),
      description: formData.description.trim() || undefined,
      permissions: formData.permissions,
      is_system_role: formData.is_system_role,
      parent_role: formData.parent_role || undefined
    }

    const newRole = await rbacStore.createRole(roleData)
    emit('created', newRole)
  } catch (error: any) {
    console.error('Error creating role:', error)
    
    // Handle specific API errors
    if (error.message.includes('already exists')) {
      errors.name = 'A role with this name already exists'
    } else {
      errors.general = error.message || 'Failed to create role'
    }
  } finally {
    isSubmitting.value = false
  }
}

// Initialize component
onMounted(async () => {
  // Ensure permissions are loaded
  if (rbacStore.permissions.length === 0) {
    await rbacStore.fetchPermissions()
  }
})
</script>

<style scoped>
/* Custom checkbox indeterminate styles */
input[type="checkbox"]:indeterminate {
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M3 8h10'/%3E%3C/svg%3E");
  background-color: currentColor;
  background-size: 16px;
  border-color: transparent;
}

/* Smooth animations for accordion */
.accordion-enter-active,
.accordion-leave-active {
  transition: all 0.3s ease;
}

.accordion-enter-from,
.accordion-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
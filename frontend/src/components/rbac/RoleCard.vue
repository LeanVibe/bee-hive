<template>
  <div 
    class="p-6 hover:bg-gray-50 transition-colors border-l-4"
    :class="[
      role.is_system_role 
        ? 'border-l-blue-500 bg-blue-50/30' 
        : 'border-l-green-500',
      !role.is_active && 'opacity-60'
    ]"
  >
    <div class="flex items-center justify-between">
      <div class="flex-1 min-w-0">
        <div class="flex items-center space-x-3 mb-2">
          <!-- Role Icon -->
          <div 
            class="w-10 h-10 rounded-lg flex items-center justify-center text-white font-semibold"
            :class="role.is_system_role ? 'bg-blue-500' : 'bg-green-500'"
          >
            {{ getInitials(role.name) }}
          </div>
          
          <!-- Role Name & Status -->
          <div class="flex-1 min-w-0">
            <div class="flex items-center space-x-2">
              <h3 class="text-lg font-semibold text-gray-900 truncate">{{ role.name }}</h3>
              
              <!-- System Role Badge -->
              <span 
                v-if="role.is_system_role"
                class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
              >
                System
              </span>
              
              <!-- Active Status -->
              <span 
                class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium"
                :class="role.is_active 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-gray-100 text-gray-800'"
              >
                {{ role.is_active ? 'Active' : 'Inactive' }}
              </span>
            </div>
            
            <!-- Description -->
            <p 
              v-if="role.description" 
              class="text-sm text-gray-600 mt-1 truncate"
              :title="role.description"
            >
              {{ role.description }}
            </p>
          </div>
        </div>
        
        <!-- Role Stats -->
        <div class="flex items-center space-x-6 text-sm text-gray-600">
          <div class="flex items-center space-x-1">
            <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <span>{{ role.user_count }} {{ role.user_count === 1 ? 'user' : 'users' }}</span>
          </div>
          
          <div class="flex items-center space-x-1">
            <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{{ permissionCount }} {{ permissionCount === 1 ? 'permission' : 'permissions' }}</span>
          </div>
          
          <div class="flex items-center space-x-1">
            <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Updated {{ formatDate(role.updated_at) }}</span>
          </div>
        </div>
        
        <!-- Permissions Preview -->
        <div v-if="role.permissions && role.permissions.length > 0" class="mt-3">
          <div class="flex flex-wrap gap-1">
            <span
              v-for="(permission, index) in previewPermissions"
              :key="typeof permission === 'string' ? permission : permission.value"
              class="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-700"
            >
              {{ formatPermission(typeof permission === 'string' ? permission : permission.value) }}
            </span>
            
            <span
              v-if="role.permissions.length > maxPreviewPermissions"
              class="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-200 text-gray-600 cursor-pointer"
              @click="showAllPermissions = !showAllPermissions"
              :title="`Click to ${showAllPermissions ? 'hide' : 'show'} all permissions`"
            >
              +{{ role.permissions.length - maxPreviewPermissions }} more
            </span>
          </div>
          
          <!-- Expanded permissions view -->
          <div 
            v-if="showAllPermissions && role.permissions.length > maxPreviewPermissions"
            class="mt-2 p-3 bg-gray-50 rounded-lg"
          >
            <div class="flex items-center justify-between mb-2">
              <h4 class="text-sm font-medium text-gray-900">All Permissions</h4>
              <button
                @click="showAllPermissions = false"
                class="text-xs text-gray-500 hover:text-gray-700"
              >
                Hide
              </button>
            </div>
            <div class="grid grid-cols-2 gap-1">
              <span
                v-for="permission in role.permissions"
                :key="typeof permission === 'string' ? permission : permission.value"
                class="text-xs text-gray-600 py-1"
              >
                • {{ formatPermission(typeof permission === 'string' ? permission : permission.value) }}
              </span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Action Buttons -->
      <div class="flex items-center space-x-2 ml-4">
        <!-- View Users Button -->
        <button
          @click="$emit('viewUsers', role)"
          class="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
          :title="`View users with ${role.name} role`"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        </button>
        
        <!-- Edit Button -->
        <button
          @click="$emit('edit', role)"
          :disabled="!canEdit"
          class="p-2 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          :title="canEdit ? `Edit ${role.name} role` : 'Cannot edit this role'"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
        </button>
        
        <!-- Delete Button -->
        <button
          @click="$emit('delete', role)"
          :disabled="!canDelete"
          class="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          :title="canDelete ? `Delete ${role.name} role` : 'Cannot delete this role'"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
        
        <!-- Dropdown Menu -->
        <div class="relative">
          <button
            @click="showDropdown = !showDropdown"
            class="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
            :class="{ 'bg-gray-50 text-gray-600': showDropdown }"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
            </svg>
          </button>
          
          <!-- Dropdown Menu -->
          <div
            v-if="showDropdown"
            v-click-outside="() => showDropdown = false"
            class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-10"
          >
            <button
              @click="duplicateRole"
              class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span>Duplicate Role</span>
            </button>
            
            <button
              @click="exportRole"
              class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>Export Role</span>
            </button>
            
            <hr class="my-1 border-gray-200">
            
            <button
              @click="viewAuditLog"
              class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>View Audit Log</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Role } from '@/stores/rbac'

interface Props {
  role: Role
}

interface Emits {
  (e: 'edit', role: Role): void
  (e: 'delete', role: Role): void
  (e: 'viewUsers', role: Role): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

// Component state
const showDropdown = ref(false)
const showAllPermissions = ref(false)
const maxPreviewPermissions = 3

// Computed properties
const permissionCount = computed(() => {
  return Array.isArray(props.role.permissions) ? props.role.permissions.length : 0
})

const previewPermissions = computed(() => {
  if (!Array.isArray(props.role.permissions)) return []
  return props.role.permissions.slice(0, maxPreviewPermissions)
})

const canEdit = computed(() => {
  // TODO: Check user permissions - for now allow editing of non-system roles
  return !props.role.is_system_role && props.role.is_active
})

const canDelete = computed(() => {
  // Cannot delete system roles or roles with users
  return !props.role.is_system_role && props.role.user_count === 0
})

// Utility functions
const getInitials = (name: string): string => {
  return name
    .split(' ')
    .map(word => word.charAt(0).toUpperCase())
    .join('')
    .substring(0, 2)
}

const formatPermission = (permission: string): string => {
  return permission
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

const formatDate = (dateString: string): string => {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
  
  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return `${diffDays} days ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`
  
  return date.toLocaleDateString()
}

// Action handlers
const duplicateRole = () => {
  console.log('Duplicate role:', props.role.name)
  showDropdown.value = false
  // TODO: Emit duplicate event or open creation modal with pre-filled data
}

const exportRole = () => {
  const roleData = {
    name: props.role.name,
    description: props.role.description,
    permissions: props.role.permissions,
    is_system_role: props.role.is_system_role
  }

  const blob = new Blob([JSON.stringify(roleData, null, 2)], { 
    type: 'application/json' 
  })
  
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `role_${props.role.name.toLowerCase().replace(/\s+/g, '_')}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  
  showDropdown.value = false
}

const viewAuditLog = () => {
  console.log('View audit log for role:', props.role.name)
  showDropdown.value = false
  // TODO: Open audit log modal filtered for this role
}

// Click outside directive
const vClickOutside = {
  mounted(el: HTMLElement, binding: any) {
    el.clickOutsideEvent = (event: Event) => {
      if (!(el === event.target || el.contains(event.target as Node))) {
        binding.value()
      }
    }
    document.addEventListener('click', el.clickOutsideEvent)
  },
  unmounted(el: HTMLElement) {
    document.removeEventListener('click', el.clickOutsideEvent)
  }
}
</script>

<style scoped>
/* Animation for expanded permissions */
.expanded-permissions {
  animation: slideDown 0.2s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
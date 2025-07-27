import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  duration?: number // 0 = persistent
  timestamp: Date
  actions?: Array<{
    label: string
    action: () => void
    style?: 'primary' | 'secondary'
  }>
}

export const useNotificationStore = defineStore('notifications', () => {
  // State
  const notifications = ref<Notification[]>([])
  const maxNotifications = ref(5)
  
  // Actions
  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    
    const newNotification: Notification = {
      id,
      timestamp: new Date(),
      duration: 5000, // Default 5 seconds
      ...notification,
    }
    
    // Add to beginning of array
    notifications.value.unshift(newNotification)
    
    // Limit number of notifications
    if (notifications.value.length > maxNotifications.value) {
      notifications.value = notifications.value.slice(0, maxNotifications.value)
    }
    
    // Auto-remove after duration (if not persistent)
    if (newNotification.duration && newNotification.duration > 0) {
      setTimeout(() => {
        removeNotification(id)
      }, newNotification.duration)
    }
    
    return id
  }
  
  const removeNotification = (id: string) => {
    const index = notifications.value.findIndex(n => n.id === id)
    if (index > -1) {
      notifications.value.splice(index, 1)
    }
  }
  
  const clearAllNotifications = () => {
    notifications.value = []
  }
  
  const markAsRead = (id: string) => {
    // In a real app, this might update a read status
    // For now, we'll just remove the notification
    removeNotification(id)
  }
  
  // Convenience methods for common notification types
  const success = (title: string, message: string, duration = 5000) => {
    return addNotification({ type: 'success', title, message, duration })
  }
  
  const error = (title: string, message: string, duration = 10000) => {
    return addNotification({ type: 'error', title, message, duration })
  }
  
  const warning = (title: string, message: string, duration = 7000) => {
    return addNotification({ type: 'warning', title, message, duration })
  }
  
  const info = (title: string, message: string, duration = 5000) => {
    return addNotification({ type: 'info', title, message, duration })
  }
  
  const persistent = (type: Notification['type'], title: string, message: string) => {
    return addNotification({ type, title, message, duration: 0 })
  }
  
  return {
    // State
    notifications,
    maxNotifications,
    
    // Actions
    addNotification,
    removeNotification,
    clearAllNotifications,
    markAsRead,
    
    // Convenience methods
    success,
    error,
    warning,
    info,
    persistent,
  }
})
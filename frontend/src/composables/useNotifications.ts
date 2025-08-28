/**
 * Notifications Composable
 * Provides a simple interface for working with notifications
 */

import { useNotificationStore } from '@/stores/notifications'

export function useNotifications() {
  const store = useNotificationStore()

  const addNotification = (notification: {
    type: 'success' | 'error' | 'warning' | 'info'
    title: string
    message: string
    duration?: number
  }) => {
    return store.addNotification(notification)
  }

  return {
    addNotification,
    removeNotification: store.removeNotification,
    clearAll: store.clearAllNotifications,
    success: store.success,
    error: store.error,
    warning: store.warning,
    info: store.info
  }
}
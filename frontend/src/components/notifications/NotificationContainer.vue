<template>
  <Teleport to="body">
    <div
      class="fixed top-4 right-4 z-50 space-y-2 max-w-sm w-full pointer-events-none"
      role="region"
      aria-label="Notifications"
    >
      <TransitionGroup
        name="notification"
        tag="div"
        class="space-y-2"
      >
        <div
          v-for="notification in notifications"
          :key="notification.id"
          class="pointer-events-auto transform transition-all duration-300 ease-in-out"
        >
          <NotificationCard
            :notification="notification"
            @close="removeNotification"
            @action="handleNotificationAction"
          />
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useNotificationStore, type Notification } from '@/stores/notifications'
import NotificationCard from './NotificationCard.vue'

const notificationStore = useNotificationStore()

// Computed
const notifications = computed(() => notificationStore.notifications)

// Actions
const removeNotification = (id: string) => {
  notificationStore.removeNotification(id)
}

const handleNotificationAction = (notification: Notification, actionIndex: number) => {
  const action = notification.actions?.[actionIndex]
  if (action) {
    action.action()
    // Optionally remove notification after action
    removeNotification(notification.id)
  }
}
</script>

<style scoped>
.notification-enter-active,
.notification-leave-active {
  transition: all 0.3s ease;
}

.notification-enter-from {
  opacity: 0;
  transform: translateX(100%);
}

.notification-leave-to {
  opacity: 0;
  transform: translateX(100%);
}

.notification-move {
  transition: transform 0.3s ease;
}
</style>
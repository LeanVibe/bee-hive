<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
    <div class="text-center">
      <div class="mb-8">
        <svg class="mx-auto h-24 w-24 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.515-.641-6.391-1.757M6 15.02C5.07 15.657 4.067 16 3 16s-2.07-.343-3-1.98M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.515-.641-6.391-1.757" />
        </svg>
      </div>
      
      <h1 class="text-4xl font-bold text-gray-900 dark:text-white mb-4">404</h1>
      <h2 class="text-2xl font-semibold text-gray-700 dark:text-gray-200 mb-4">Page Not Found</h2>
      <p class="text-gray-600 dark:text-gray-400 mb-8 max-w-md">
        The page you're looking for doesn't exist. It might have been moved, deleted, or you entered the wrong URL.
      </p>
      
      <div class="space-y-4">
        <router-link 
          to="/" 
          class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
        >
          <svg class="mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>
          Go to Dashboard
        </router-link>
        
        <div class="text-sm text-gray-500 dark:text-gray-400">
          Or try one of these pages:
        </div>
        
        <div class="flex justify-center space-x-4">
          <router-link 
            to="/metrics" 
            class="text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Metrics
          </router-link>
          <router-link 
            to="/events" 
            class="text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Events
          </router-link>
          <router-link 
            to="/settings" 
            class="text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Settings
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// No additional logic needed for 404 page
</script>
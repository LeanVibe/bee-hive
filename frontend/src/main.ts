import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import './assets/styles/main.css'

// Create Vue application
const app = createApp(App)

// Install plugins
app.use(createPinia())
app.use(router)

// Global error handler
app.config.errorHandler = (err, _vm, info) => {
  console.error('Global error:', err, info)
  // In production, send to error tracking service
}

// Mount application
app.mount('#app')
import { openDB, IDBPDatabase, IDBPTransaction } from 'idb'
import { EventEmitter } from '../utils/event-emitter'

export interface CachedData {
  id: string
  type: string
  data: any
  timestamp: number
  expiresAt?: number
  version: number
}

export interface SyncOperation {
  id: string
  type: 'create' | 'update' | 'delete'
  resource: string
  data: any
  timestamp: number
  retryCount: number
  maxRetries: number
  lastError?: string
}

export interface OfflineTask {
  id: string
  title: string
  description?: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  priority: 'low' | 'medium' | 'high'
  agent_id?: string
  created_at: number
  updated_at: number
  synced: boolean
}

export class OfflineService extends EventEmitter {
  private static instance: OfflineService
  private db: IDBPDatabase | null = null
  private syncQueue: SyncOperation[] = []
  private isOnline: boolean = navigator.onLine
  private syncInProgress: boolean = false
  private readonly dbName = 'AgentHiveOfflineDB'
  private readonly dbVersion = 1
  private readonly maxCacheAge = 24 * 60 * 60 * 1000 // 24 hours
  private readonly maxRetries = 3
  
  static getInstance(): OfflineService {
    if (!OfflineService.instance) {
      OfflineService.instance = new OfflineService()
    }
    return OfflineService.instance
  }
  
  constructor() {
    super()
    
    // Listen for online/offline events
    window.addEventListener('online', this.handleOnline.bind(this))
    window.addEventListener('offline', this.handleOffline.bind(this))
  }
  
  async initialize(): Promise<void> {
    try {
      console.log('💾 Initializing offline service...')
      
      await this.initializeDB()
      await this.loadSyncQueue()
      await this.cleanupExpiredData()
      
      // Start background sync if online
      if (this.isOnline) {
        this.startBackgroundSync()
      }
      
      console.log('✅ Offline service initialized')
      
    } catch (error) {
      console.error('❌ Failed to initialize offline service:', error)
      throw error
    }
  }
  
  private async initializeDB(): Promise<void> {
    this.db = await openDB(this.dbName, this.dbVersion, {
      upgrade(db, oldVersion, newVersion, transaction) {
        // Create object stores
        if (!db.objectStoreNames.contains('cache')) {
          const cacheStore = db.createObjectStore('cache', { keyPath: 'id' })
          cacheStore.createIndex('type', 'type')
          cacheStore.createIndex('timestamp', 'timestamp')
          cacheStore.createIndex('expiresAt', 'expiresAt')
        }
        
        if (!db.objectStoreNames.contains('sync_queue')) {
          const syncStore = db.createObjectStore('sync_queue', { keyPath: 'id' })
          syncStore.createIndex('type', 'type')
          syncStore.createIndex('resource', 'resource')
          syncStore.createIndex('timestamp', 'timestamp')
        }
        
        if (!db.objectStoreNames.contains('tasks')) {
          const tasksStore = db.createObjectStore('tasks', { keyPath: 'id' })
          tasksStore.createIndex('status', 'status')
          tasksStore.createIndex('priority', 'priority')
          tasksStore.createIndex('agent_id', 'agent_id')
          tasksStore.createIndex('synced', 'synced')
          tasksStore.createIndex('updated_at', 'updated_at')
        }
        
        if (!db.objectStoreNames.contains('agents')) {
          const agentsStore = db.createObjectStore('agents', { keyPath: 'id' })
          agentsStore.createIndex('status', 'status')
          agentsStore.createIndex('updated_at', 'updated_at')
        }
        
        if (!db.objectStoreNames.contains('events')) {
          const eventsStore = db.createObjectStore('events', { keyPath: 'id' })
          eventsStore.createIndex('event_type', 'event_type')
          eventsStore.createIndex('agent_id', 'agent_id')
          eventsStore.createIndex('timestamp', 'timestamp')
        }
      },
    })
  }
  
  // Cache management
  async cacheData(id: string, type: string, data: any, ttl?: number): Promise<void> {
    if (!this.db) return
    
    const cachedData: CachedData = {
      id,
      type,
      data,
      timestamp: Date.now(),
      expiresAt: ttl ? Date.now() + ttl : undefined,
      version: 1
    }
    
    await this.db.put('cache', cachedData)
    this.emit('data_cached', { id, type })
  }
  
  async getCachedData(id: string): Promise<any | null> {
    if (!this.db) return null
    
    const cachedData = await this.db.get('cache', id)
    if (!cachedData) return null
    
    // Check if expired
    if (cachedData.expiresAt && Date.now() > cachedData.expiresAt) {
      await this.db.delete('cache', id)
      return null
    }
    
    return cachedData.data
  }
  
  async getCachedDataByType(type: string): Promise<any[]> {
    if (!this.db) return []
    
    const tx = this.db.transaction('cache', 'readonly')
    const index = tx.store.index('type')
    const items = await index.getAll(type)
    
    // Filter out expired items
    const now = Date.now()
    const validItems = items.filter(item => {
      if (item.expiresAt && now > item.expiresAt) {
        // Schedule for deletion
        this.db?.delete('cache', item.id)
        return false
      }
      return true
    })
    
    return validItems.map(item => item.data)
  }
  
  async clearCache(type?: string): Promise<void> {
    if (!this.db) return
    
    if (type) {
      const tx = this.db.transaction('cache', 'readwrite')
      const index = tx.store.index('type')
      const items = await index.getAllKeys(type)
      
      for (const key of items) {
        await tx.store.delete(key)
      }
    } else {
      await this.db.clear('cache')
    }
    
    this.emit('cache_cleared', { type })
  }
  
  // Task management (offline-first)
  async saveTasks(tasks: OfflineTask[]): Promise<void> {
    if (!this.db) return
    
    const tx = this.db.transaction('tasks', 'readwrite')
    
    for (const task of tasks) {
      await tx.store.put({
        ...task,
        updated_at: Date.now(),
        synced: this.isOnline
      })
    }
    
    await tx.done
    this.emit('tasks_saved', tasks)
  }
  
  async getTask(id: string): Promise<OfflineTask | null> {
    if (!this.db) return null
    return await this.db.get('tasks', id)
  }
  
  async getTasks(filters?: {
    status?: string
    priority?: string
    agent_id?: string
    synced?: boolean
  }): Promise<OfflineTask[]> {
    if (!this.db) return []
    
    let tasks = await this.db.getAll('tasks')
    
    if (filters) {
      tasks = tasks.filter(task => {
        if (filters.status && task.status !== filters.status) return false
        if (filters.priority && task.priority !== filters.priority) return false
        if (filters.agent_id && task.agent_id !== filters.agent_id) return false
        if (filters.synced !== undefined && task.synced !== filters.synced) return false
        return true
      })
    }
    
    return tasks.sort((a, b) => b.updated_at - a.updated_at)
  }
  
  async updateTask(id: string, updates: Partial<OfflineTask>): Promise<void> {
    if (!this.db) return
    
    const existingTask = await this.db.get('tasks', id)
    if (!existingTask) {
      throw new Error(`Task ${id} not found`)
    }
    
    const updatedTask: OfflineTask = {
      ...existingTask,
      ...updates,
      updated_at: Date.now(),
      synced: false // Mark as needing sync
    }
    
    await this.db.put('tasks', updatedTask)
    
    // Queue for sync
    await this.queueSync('update', 'tasks', updatedTask)
    
    this.emit('task_updated', updatedTask)
  }
  
  async deleteTask(id: string): Promise<void> {
    if (!this.db) return
    
    const task = await this.db.get('tasks', id)
    if (!task) return
    
    await this.db.delete('tasks', id)
    
    // Queue for sync
    await this.queueSync('delete', 'tasks', { id })
    
    this.emit('task_deleted', { id })
  }
  
  // Sync queue management
  async queueSync(type: 'create' | 'update' | 'delete', resource: string, data: any): Promise<void> {
    if (!this.db) return
    
    const operation: SyncOperation = {
      id: crypto.randomUUID(),
      type,
      resource,
      data,
      timestamp: Date.now(),
      retryCount: 0,
      maxRetries: this.maxRetries
    }
    
    await this.db.put('sync_queue', operation)
    this.syncQueue.push(operation)
    
    // Attempt sync if online
    if (this.isOnline && !this.syncInProgress) {
      this.startBackgroundSync()
    }
    
    this.emit('sync_queued', operation)
  }
  
  private async loadSyncQueue(): Promise<void> {
    if (!this.db) return
    
    this.syncQueue = await this.db.getAll('sync_queue')
    console.log(`📤 Loaded ${this.syncQueue.length} operations in sync queue`)
  }
  
  private async startBackgroundSync(): Promise<void> {
    if (this.syncInProgress || !this.isOnline) return
    
    this.syncInProgress = true
    console.log('🔄 Starting background sync...')
    
    try {
      const operations = [...this.syncQueue]
      
      for (const operation of operations) {
        try {
          await this.executeSync(operation)
          
          // Remove from queue and database
          this.syncQueue = this.syncQueue.filter(op => op.id !== operation.id)
          await this.db?.delete('sync_queue', operation.id)
          
          this.emit('sync_success', operation)
          
        } catch (error) {
          console.error(`Sync operation ${operation.id} failed:`, error)
          
          // Increment retry count
          operation.retryCount++
          operation.lastError = error instanceof Error ? error.message : 'Unknown error'
          
          if (operation.retryCount >= operation.maxRetries) {
            // Remove failed operation
            this.syncQueue = this.syncQueue.filter(op => op.id !== operation.id)
            await this.db?.delete('sync_queue', operation.id)
            this.emit('sync_failed', operation)
          } else {
            // Update retry count in database
            await this.db?.put('sync_queue', operation)
            this.emit('sync_retry', operation)
          }
        }
      }
      
      console.log('✅ Background sync completed')
      
    } finally {
      this.syncInProgress = false
    }
  }
  
  private async executeSync(operation: SyncOperation): Promise<void> {
    const { type, resource, data } = operation
    
    switch (resource) {
      case 'tasks':
        await this.syncTask(type, data)
        break
      case 'agents':
        await this.syncAgent(type, data)
        break
      default:
        throw new Error(`Unknown sync resource: ${resource}`)
    }
  }
  
  private async syncTask(type: 'create' | 'update' | 'delete', data: any): Promise<void> {
    const url = `/api/v1/tasks${type === 'create' ? '' : `/${data.id}`}`
    const method = type === 'create' ? 'POST' : type === 'update' ? 'PUT' : 'DELETE'
    
    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
      },
      body: type !== 'delete' ? JSON.stringify(data) : undefined
    })
    
    if (!response.ok) {
      throw new Error(`Task sync failed: ${response.status} ${response.statusText}`)
    }
    
    // Mark as synced in local database
    if (type !== 'delete' && this.db) {
      const task = await this.db.get('tasks', data.id)
      if (task) {
        task.synced = true
        await this.db.put('tasks', task)
      }
    }
  }
  
  private async syncAgent(type: 'create' | 'update' | 'delete', data: any): Promise<void> {
    // Similar implementation for agent sync
    console.log('Agent sync not implemented yet:', type, data)
  }
  
  private async cleanupExpiredData(): Promise<void> {
    if (!this.db) return
    
    const now = Date.now()
    const tx = this.db.transaction('cache', 'readwrite')
    const index = tx.store.index('expiresAt')
    
    // Get all expired items
    const expiredItems = await index.getAll(IDBKeyRange.upperBound(now))
    
    // Delete expired items
    for (const item of expiredItems) {
      await tx.store.delete(item.id)
    }
    
    await tx.done
    
    if (expiredItems.length > 0) {
      console.log(`🧹 Cleaned up ${expiredItems.length} expired cache entries`)
    }
  }
  
  private handleOnline(): void {
    console.log('🌐 App is online')
    this.isOnline = true
    this.emit('online')
    
    // Start sync
    if (!this.syncInProgress) {
      this.startBackgroundSync()
    }
  }
  
  private handleOffline(): void {
    console.log('📱 App is offline')
    this.isOnline = false
    this.emit('offline')
  }
  
  // Public getters
  isOnlineMode(): boolean {
    return this.isOnline
  }
  
  getSyncQueueLength(): number {
    return this.syncQueue.length
  }
  
  isSyncing(): boolean {
    return this.syncInProgress
  }
  
  // Utility methods
  async getStorageUsage(): Promise<{ used: number; quota: number }> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate()
      return {
        used: estimate.usage || 0,
        quota: estimate.quota || 0
      }
    }
    
    return { used: 0, quota: 0 }
  }
  
  async clearAllData(): Promise<void> {
    if (!this.db) return
    
    const storeNames = ['cache', 'sync_queue', 'tasks', 'agents', 'events']
    
    for (const storeName of storeNames) {
      await this.db.clear(storeName)
    }
    
    this.syncQueue = []
    this.emit('data_cleared')
  }
  
  // Export/import for debugging
  async exportData(): Promise<any> {
    if (!this.db) return {}
    
    const data: any = {}
    const storeNames = ['cache', 'sync_queue', 'tasks', 'agents', 'events']
    
    for (const storeName of storeNames) {
      data[storeName] = await this.db.getAll(storeName)
    }
    
    return data
  }
  
  async importData(data: any): Promise<void> {
    if (!this.db) return
    
    for (const [storeName, items] of Object.entries(data)) {
      if (Array.isArray(items)) {
        const tx = this.db.transaction(storeName, 'readwrite')
        for (const item of items) {
          await tx.store.put(item)
        }
        await tx.done
      }
    }
    
    // Reload sync queue
    await this.loadSyncQueue()
    
    this.emit('data_imported')
  }
}
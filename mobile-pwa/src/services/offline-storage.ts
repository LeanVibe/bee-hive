import { Task } from '../types/task'
import { AgentStatus } from '../components/dashboard/agent-health-panel'
import { TimelineEvent } from '../components/dashboard/event-timeline'

export interface OfflineAction {
  id: string
  type: 'task-update' | 'task-create' | 'task-delete'
  data: any
  timestamp: string
  synced: boolean
}

export class OfflineStorageService {
  private static instance: OfflineStorageService
  private db: IDBDatabase | null = null
  private readonly dbName = 'AgentHiveOfflineDB'
  private readonly dbVersion = 1
  
  static getInstance(): OfflineStorageService {
    if (!OfflineStorageService.instance) {
      OfflineStorageService.instance = new OfflineStorageService()
    }
    return OfflineStorageService.instance
  }
  
  async initialize(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!('indexedDB' in window)) {
        console.warn('IndexedDB not supported')
        resolve()
        return
      }
      
      const request = indexedDB.open(this.dbName, this.dbVersion)
      
      request.onerror = () => {
        console.error('Failed to open IndexedDB:', request.error)
        reject(request.error)
      }
      
      request.onsuccess = () => {
        this.db = request.result
        console.log('✅ IndexedDB initialized')
        resolve()
      }
      
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result
        
        // Tasks store
        if (!db.objectStoreNames.contains('tasks')) {
          const tasksStore = db.createObjectStore('tasks', { keyPath: 'id' })
          tasksStore.createIndex('status', 'status', { unique: false })
          tasksStore.createIndex('agent', 'agent', { unique: false })
          tasksStore.createIndex('updatedAt', 'updatedAt', { unique: false })
        }
        
        // Agents store
        if (!db.objectStoreNames.contains('agents')) {
          const agentsStore = db.createObjectStore('agents', { keyPath: 'id' })
          agentsStore.createIndex('status', 'status', { unique: false })
          agentsStore.createIndex('lastSeen', 'lastSeen', { unique: false })
        }
        
        // Events store
        if (!db.objectStoreNames.contains('events')) {
          const eventsStore = db.createObjectStore('events', { keyPath: 'id' })
          eventsStore.createIndex('type', 'type', { unique: false })
          eventsStore.createIndex('agent', 'agent', { unique: false })
          eventsStore.createIndex('timestamp', 'timestamp', { unique: false })
        }
        
        // Offline actions store
        if (!db.objectStoreNames.contains('offline_actions')) {
          const actionsStore = db.createObjectStore('offline_actions', { keyPath: 'id' })
          actionsStore.createIndex('type', 'type', { unique: false })
          actionsStore.createIndex('synced', 'synced', { unique: false })
          actionsStore.createIndex('timestamp', 'timestamp', { unique: false })
        }
        
        // App settings store
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'key' })
        }
        
        console.log('✅ IndexedDB schema upgraded')
      }
    })
  }
  
  // Tasks
  async cacheTasks(tasks: Task[]): Promise<void> {
    if (!this.db) return
    
    const transaction = this.db.transaction(['tasks'], 'readwrite')
    const store = transaction.objectStore('tasks')
    
    // Clear existing tasks
    await this.promisifyRequest(store.clear())
    
    // Add new tasks
    for (const task of tasks) {
      await this.promisifyRequest(store.add(task))
    }
    
    console.log(`✅ Cached ${tasks.length} tasks`)
  }
  
  async getTasks(): Promise<Task[]> {
    if (!this.db) return []
    
    const transaction = this.db.transaction(['tasks'], 'readonly')
    const store = transaction.objectStore('tasks')
    const request = store.getAll()
    
    const tasks = await this.promisifyRequest(request) as Task[]
    return tasks || []
  }
  
  // Agents
  async cacheAgents(agents: AgentStatus[]): Promise<void> {
    if (!this.db) return
    
    const transaction = this.db.transaction(['agents'], 'readwrite')
    const store = transaction.objectStore('agents')
    
    // Clear existing agents
    await this.promisifyRequest(store.clear())
    
    // Add new agents
    for (const agent of agents) {
      await this.promisifyRequest(store.add(agent))
    }
    
    console.log(`✅ Cached ${agents.length} agents`)
  }
  
  async getAgents(): Promise<AgentStatus[]> {
    if (!this.db) return []
    
    const transaction = this.db.transaction(['agents'], 'readonly')
    const store = transaction.objectStore('agents')
    const request = store.getAll()
    
    const agents = await this.promisifyRequest(request) as AgentStatus[]
    return agents || []
  }
  
  // Events
  async cacheEvents(events: TimelineEvent[]): Promise<void> {
    if (!this.db) return
    
    const transaction = this.db.transaction(['events'], 'readwrite')
    const store = transaction.objectStore('events')
    
    // Keep only the most recent 200 events
    const sortedEvents = events
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 200)
    
    // Clear existing events
    await this.promisifyRequest(store.clear())
    
    // Add new events
    for (const event of sortedEvents) {
      await this.promisifyRequest(store.add(event))
    }
    
    console.log(`✅ Cached ${sortedEvents.length} events`)
  }
  
  async getEvents(): Promise<TimelineEvent[]> {
    if (!this.db) return []
    
    const transaction = this.db.transaction(['events'], 'readonly')
    const store = transaction.objectStore('events')
    const request = store.getAll()
    
    const events = await this.promisifyRequest(request) as TimelineEvent[]
    return events || []
  }
  
  // Offline Actions
  async queueOfflineAction(action: Omit<OfflineAction, 'id' | 'timestamp' | 'synced'>): Promise<void> {
    if (!this.db) return
    
    const offlineAction: OfflineAction = {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      synced: false,
      ...action
    }
    
    const transaction = this.db.transaction(['offline_actions'], 'readwrite')
    const store = transaction.objectStore('offline_actions')
    
    await this.promisifyRequest(store.add(offlineAction))
    console.log('✅ Queued offline action:', action.type)
    
    // Dispatch event for PWA sync
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'queue-offline-action',
        data: offlineAction
      })
    }
  }
  
  async getOfflineActions(): Promise<OfflineAction[]> {
    if (!this.db) return []
    
    const transaction = this.db.transaction(['offline_actions'], 'readonly')
    const store = transaction.objectStore('offline_actions')
    const index = store.index('synced')
    const request = index.getAll(false) // Get unsynced actions
    
    const actions = await this.promisifyRequest(request) as OfflineAction[]
    return actions || []
  }
  
  async markActionSynced(actionId: string): Promise<void> {
    if (!this.db) return
    
    const transaction = this.db.transaction(['offline_actions'], 'readwrite')
    const store = transaction.objectStore('offline_actions')
    
    const action = await this.promisifyRequest(store.get(actionId)) as OfflineAction
    if (action) {
      action.synced = true
      await this.promisifyRequest(store.put(action))
    }
  }
  
  async clearSyncedActions(): Promise<void> {
    if (!this.db) return
    
    const transaction = this.db.transaction(['offline_actions'], 'readwrite')
    const store = transaction.objectStore('offline_actions')
    const index = store.index('synced')
    const request = index.openCursor(true) // Get synced actions
    
    const cursor = await this.promisifyRequest(request)
    if (cursor) {
      await this.promisifyRequest(store.delete(cursor.primaryKey))
      cursor.continue()
    }
  }
  
  // Settings
  async setSetting(key: string, value: any): Promise<void> {
    if (!this.db) return
    
    const transaction = this.db.transaction(['settings'], 'readwrite')
    const store = transaction.objectStore('settings')
    
    await this.promisifyRequest(store.put({ key, value }))
  }
  
  async getSetting(key: string, defaultValue?: any): Promise<any> {
    if (!this.db) return defaultValue
    
    const transaction = this.db.transaction(['settings'], 'readonly')
    const store = transaction.objectStore('settings')
    const request = store.get(key)
    
    const result = await this.promisifyRequest(request)
    return result ? result.value : defaultValue
  }
  
  // Utility methods
  async clearAllData(): Promise<void> {
    if (!this.db) return
    
    const storeNames = ['tasks', 'agents', 'events', 'offline_actions', 'settings']
    const transaction = this.db.transaction(storeNames, 'readwrite')
    
    for (const storeName of storeNames) {
      const store = transaction.objectStore(storeName)
      await this.promisifyRequest(store.clear())
    }
    
    console.log('✅ Cleared all offline data')
  }
  
  async getStorageStats(): Promise<{
    tasks: number
    agents: number
    events: number
    actions: number
    settings: number
  }> {
    if (!this.db) {
      return { tasks: 0, agents: 0, events: 0, actions: 0, settings: 0 }
    }
    
    const storeNames = ['tasks', 'agents', 'events', 'offline_actions', 'settings']
    const transaction = this.db.transaction(storeNames, 'readonly')
    
    const stats = {
      tasks: 0,
      agents: 0,
      events: 0,
      actions: 0,
      settings: 0
    }
    
    stats.tasks = await this.promisifyRequest(transaction.objectStore('tasks').count())
    stats.agents = await this.promisifyRequest(transaction.objectStore('agents').count())
    stats.events = await this.promisifyRequest(transaction.objectStore('events').count())
    stats.actions = await this.promisifyRequest(transaction.objectStore('offline_actions').count())
    stats.settings = await this.promisifyRequest(transaction.objectStore('settings').count())
    
    return stats
  }
  
  private promisifyRequest<T = any>(request: IDBRequest<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
  }
}
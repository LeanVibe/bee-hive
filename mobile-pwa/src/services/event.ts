/**
 * Event Service for LeanVibe Agent Hive
 * 
 * Provides real-time system event monitoring including:
 * - System event streaming and timeline
 * - Event filtering and search
 * - WebSocket real-time updates (with fallback to polling)
 * - Event acknowledgment and management
 * - Activity timeline for dashboard
 * - Event persistence for offline viewing
 */

import { BaseService } from './base-service';
import {
  EventType,
  EventSeverity
} from '../types/api';
import type {
  SystemEvent,
  EventFilters,
  WebSocketMessage,
  WebSocketEvent,
  ServiceConfig,
  Subscription,
  EventListener
} from '../types/api';

export interface EventTimeline {
  events: SystemEvent[];
  hasMore: boolean;
  lastEventId?: string;
  totalEvents: number;
}

export interface EventStatistics {
  total: number;
  bySeverity: Record<EventSeverity, number>;
  byType: Record<EventType, number>;
  unacknowledged: number;
  recentActivity: {
    last24h: number;
    lastHour: number;
    last5min: number;
  };
}

export interface ActivitySummary {
  timestamp: string;
  events: SystemEvent[];
  agentActivity: {
    agentId: string;
    agentName: string;
    eventCount: number;
    lastActivity: string;
  }[];
  taskActivity: {
    taskId: string;
    taskTitle: string;
    eventCount: number;
    lastActivity: string;
  }[];
}

export class EventService extends BaseService {
  private systemEvents: Map<string, SystemEvent> = new Map();
  private eventTimeline: SystemEvent[] = [];
  private websocket: WebSocket | null = null;
  private pollingStopFn: (() => void) | null = null;
  private maxTimelineSize = 1000;
  private connectionRetries = 0;
  private maxRetries = 5;

  constructor(config: Partial<ServiceConfig> = {}) {
    super({
      pollingInterval: 5000, // 5 seconds fallback polling
      cacheTimeout: 30000, // 30 second cache for events
      ...config
    });
  }

  // ===== EVENT RETRIEVAL =====

  /**
   * Get recent system events with filtering
   */
  async getRecentEvents(
    filters: EventFilters = {},
    limit = 50,
    offset = 0
  ): Promise<EventTimeline> {
    try {
      // Since we don't have a direct events API, we'll simulate with system activity
      // In a real implementation, this would connect to /api/v1/events
      const mockEvents = this.generateSimulatedEvents(limit);
      
      // Apply filters
      let filteredEvents = mockEvents;
      
      if (filters.type && filters.type.length > 0) {
        filteredEvents = filteredEvents.filter(event => 
          filters.type!.includes(event.type)
        );
      }
      
      if (filters.severity && filters.severity.length > 0) {
        filteredEvents = filteredEvents.filter(event => 
          filters.severity!.includes(event.severity)
        );
      }
      
      if (filters.agent_id) {
        filteredEvents = filteredEvents.filter(event => 
          event.agent_id === filters.agent_id
        );
      }
      
      if (filters.acknowledged !== undefined) {
        filteredEvents = filteredEvents.filter(event => 
          event.acknowledged === filters.acknowledged
        );
      }
      
      if (filters.start_date || filters.end_date) {
        filteredEvents = filteredEvents.filter(event => {
          const eventTime = new Date(event.timestamp);
          if (filters.start_date && eventTime < new Date(filters.start_date)) {
            return false;
          }
          if (filters.end_date && eventTime > new Date(filters.end_date)) {
            return false;
          }
          return true;
        });
      }

      // Apply pagination
      const paginatedEvents = filteredEvents.slice(offset, offset + limit);
      
      // Update local cache
      paginatedEvents.forEach(event => {
        this.systemEvents.set(event.id, event);
      });

      const timeline: EventTimeline = {
        events: paginatedEvents,
        hasMore: offset + limit < filteredEvents.length,
        lastEventId: paginatedEvents[paginatedEvents.length - 1]?.id,
        totalEvents: filteredEvents.length
      };

      this.emit('eventsLoaded', timeline);
      return timeline;

    } catch (error) {
      this.emit('eventsLoadFailed', { filters, error });
      throw error;
    }
  }

  /**
   * Get specific event by ID
   */
  async getEvent(eventId: string): Promise<SystemEvent> {
    // Check local cache first
    const cached = this.systemEvents.get(eventId);
    if (cached) {
      return cached;
    }

    try {
      // In real implementation: await this.get<SystemEvent>(`/api/v1/events/${eventId}`)
      throw new Error('Event not found in local cache');
    } catch (error) {
      this.emit('eventLoadFailed', { eventId, error });
      throw error;
    }
  }

  /**
   * Acknowledge an event
   */
  async acknowledgeEvent(eventId: string): Promise<SystemEvent> {
    try {
      // In real implementation: await this.post<SystemEvent>(`/api/v1/events/${eventId}/acknowledge`)
      
      // Update local cache
      const event = this.systemEvents.get(eventId);
      if (event) {
        event.acknowledged = true;
        this.systemEvents.set(eventId, event);
        this.emit('eventAcknowledged', event);
        return event;
      }
      
      throw new Error('Event not found');
    } catch (error) {
      this.emit('eventAcknowledgeFailed', { eventId, error });
      throw error;
    }
  }

  /**
   * Acknowledge multiple events
   */
  async acknowledgeEvents(eventIds: string[]): Promise<SystemEvent[]> {
    const acknowledgedEvents: SystemEvent[] = [];
    
    for (const eventId of eventIds) {
      try {
        const event = await this.acknowledgeEvent(eventId);
        acknowledgedEvents.push(event);
      } catch (error) {
        // Continue with other events even if one fails
        console.warn(`Failed to acknowledge event ${eventId}:`, error);
      }
    }
    
    this.emit('eventsAcknowledged', acknowledgedEvents);
    return acknowledgedEvents;
  }

  // ===== REAL-TIME EVENT STREAMING =====

  /**
   * Start real-time event monitoring (WebSocket preferred, polling fallback)
   */
  startRealtimeMonitoring(): void {
    this.stopRealtimeMonitoring();
    
    // Try WebSocket first
    this.connectWebSocket();
    
    // Start polling as fallback
    this.startPollingFallback();
    
    this.emit('realtimeMonitoringStarted');
  }

  /**
   * Stop real-time event monitoring
   */
  stopRealtimeMonitoring(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    if (this.pollingStopFn) {
      this.pollingStopFn();
      this.pollingStopFn = null;
    }
    
    this.connectionRetries = 0;
    this.emit('realtimeMonitoringStopped');
  }

  /**
   * Check if real-time monitoring is active
   */
  isRealtimeMonitoring(): boolean {
    return this.websocket !== null || this.pollingStopFn !== null;
  }

  // ===== EVENT ANALYTICS =====

  /**
   * Get event statistics
   */
  getEventStatistics(): EventStatistics {
    const events = Array.from(this.systemEvents.values());
    const now = Date.now();
    const hour = 60 * 60 * 1000;
    const day = 24 * hour;
    const fiveMin = 5 * 60 * 1000;

    const stats: EventStatistics = {
      total: events.length,
      bySeverity: {} as Record<EventSeverity, number>,
      byType: {} as Record<EventType, number>,
      unacknowledged: 0,
      recentActivity: {
        last24h: 0,
        lastHour: 0,
        last5min: 0
      }
    };

    // Initialize counters
    Object.values(EventSeverity).forEach(severity => {
      stats.bySeverity[severity] = 0;
    });
    Object.values(EventType).forEach(type => {
      stats.byType[type] = 0;
    });

    events.forEach(event => {
      stats.bySeverity[event.severity]++;
      stats.byType[event.type]++;
      
      if (!event.acknowledged) {
        stats.unacknowledged++;
      }

      const eventTime = new Date(event.timestamp).getTime();
      if (now - eventTime < day) {
        stats.recentActivity.last24h++;
      }
      if (now - eventTime < hour) {
        stats.recentActivity.lastHour++;
      }
      if (now - eventTime < fiveMin) {
        stats.recentActivity.last5min++;
      }
    });

    return stats;
  }

  /**
   * Get activity summary for dashboard
   */
  getActivitySummary(hours = 24): ActivitySummary {
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    const recentEvents = Array.from(this.systemEvents.values())
      .filter(event => new Date(event.timestamp).getTime() > cutoff)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    // Group by agent
    const agentActivity = new Map<string, {
      agentId: string;
      agentName: string;
      eventCount: number;
      lastActivity: string;
    }>();

    // Group by task
    const taskActivity = new Map<string, {
      taskId: string;
      taskTitle: string;
      eventCount: number;
      lastActivity: string;
    }>();

    recentEvents.forEach(event => {
      if (event.agent_id) {
        const existing = agentActivity.get(event.agent_id) || {
          agentId: event.agent_id,
          agentName: event.source || `Agent ${event.agent_id}`,
          eventCount: 0,
          lastActivity: event.timestamp
        };
        existing.eventCount++;
        if (new Date(event.timestamp) > new Date(existing.lastActivity)) {
          existing.lastActivity = event.timestamp;
        }
        agentActivity.set(event.agent_id, existing);
      }

      if (event.task_id) {
        const existing = taskActivity.get(event.task_id) || {
          taskId: event.task_id,
          taskTitle: event.title || `Task ${event.task_id}`,
          eventCount: 0,
          lastActivity: event.timestamp
        };
        existing.eventCount++;
        if (new Date(event.timestamp) > new Date(existing.lastActivity)) {
          existing.lastActivity = event.timestamp;
        }
        taskActivity.set(event.task_id, existing);
      }
    });

    return {
      timestamp: new Date().toISOString(),
      events: recentEvents.slice(0, 50), // Latest 50 events
      agentActivity: Array.from(agentActivity.values())
        .sort((a, b) => b.eventCount - a.eventCount)
        .slice(0, 10), // Top 10 most active agents
      taskActivity: Array.from(taskActivity.values())
        .sort((a, b) => b.eventCount - a.eventCount)
        .slice(0, 10) // Top 10 most active tasks
    };
  }

  // ===== PRIVATE METHODS =====

  private connectWebSocket(): void {
    try {
      const wsUrl = this.config.baseUrl.replace('http', 'ws') + '/ws/events';
      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        this.connectionRetries = 0;
        this.emit('websocketConnected');
      };

      this.websocket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleWebSocketMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.websocket.onclose = () => {
        this.websocket = null;
        this.emit('websocketDisconnected');
        
        // Attempt reconnection if under retry limit
        if (this.connectionRetries < this.maxRetries) {
          this.connectionRetries++;
          setTimeout(() => {
            if (this.isRealtimeMonitoring()) {
              this.connectWebSocket();
            }
          }, Math.pow(2, this.connectionRetries) * 1000); // Exponential backoff
        }
      };

      this.websocket.onerror = (error) => {
        this.emit('websocketError', error);
      };

    } catch (error) {
      this.emit('websocketConnectionFailed', error);
    }
  }

  private handleWebSocketMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'event':
        const eventMessage = message as WebSocketEvent;
        this.handleNewEvent(eventMessage.data);
        break;
      default:
        // Handle other message types as needed
        break;
    }
  }

  private handleNewEvent(event: SystemEvent): void {
    // Add to local cache
    this.systemEvents.set(event.id, event);
    
    // Add to timeline
    this.eventTimeline.unshift(event);
    if (this.eventTimeline.length > this.maxTimelineSize) {
      this.eventTimeline.pop();
    }
    
    // Emit event
    this.emit('newEvent', event);
    
    // Emit severity-specific events
    if (event.severity === EventSeverity.CRITICAL) {
      this.emit('criticalEvent', event);
    } else if (event.severity === EventSeverity.ERROR) {
      this.emit('errorEvent', event);
    }
  }

  private startPollingFallback(): void {
    this.pollingStopFn = this.startPolling(async () => {
      try {
        // Poll for recent events as fallback
        await this.getRecentEvents({}, 20, 0);
      } catch (error) {
        // Polling errors are handled by base class
      }
    }, this.config.pollingInterval);
  }

  private generateSimulatedEvents(count: number): SystemEvent[] {
    const events: SystemEvent[] = [];
    const types = Object.values(EventType);
    const severities = Object.values(EventSeverity);
    
    for (let i = 0; i < count; i++) {
      const type = types[Math.floor(Math.random() * types.length)];
      const severity = severities[Math.floor(Math.random() * severities.length)];
      const timestamp = new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString();
      
      const event: SystemEvent = {
        id: `event_${Date.now()}_${i}`,
        type,
        severity,
        title: this.getEventTitle(type),
        description: this.getEventDescription(type),
        source: `system`,
        agent_id: Math.random() > 0.5 ? `agent_${Math.floor(Math.random() * 5)}` : undefined,
        task_id: Math.random() > 0.7 ? `task_${Math.floor(Math.random() * 10)}` : undefined,
        data: {
          component: 'simulation',
          action: type
        },
        timestamp,
        acknowledged: Math.random() > 0.3
      };
      
      events.push(event);
    }
    
    return events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  private getEventTitle(type: EventType): string {
    const titles: Record<EventType, string> = {
      [EventType.AGENT_ACTIVATED]: 'Agent Activated',
      [EventType.AGENT_DEACTIVATED]: 'Agent Deactivated',
      [EventType.TASK_CREATED]: 'Task Created',
      [EventType.TASK_ASSIGNED]: 'Task Assigned',
      [EventType.TASK_STARTED]: 'Task Started',
      [EventType.TASK_COMPLETED]: 'Task Completed',
      [EventType.TASK_FAILED]: 'Task Failed',
      [EventType.SYSTEM_ERROR]: 'System Error',
      [EventType.PERFORMANCE_ALERT]: 'Performance Alert',
      [EventType.HEALTH_CHECK]: 'Health Check'
    };
    return titles[type] || 'System Event';
  }

  private getEventDescription(type: EventType): string {
    const descriptions: Record<EventType, string> = {
      [EventType.AGENT_ACTIVATED]: 'New agent has been activated and is ready for work',
      [EventType.AGENT_DEACTIVATED]: 'Agent has been deactivated and stopped',
      [EventType.TASK_CREATED]: 'New development task has been created',
      [EventType.TASK_ASSIGNED]: 'Task has been assigned to an agent',
      [EventType.TASK_STARTED]: 'Agent has started working on assigned task',
      [EventType.TASK_COMPLETED]: 'Task has been completed successfully',
      [EventType.TASK_FAILED]: 'Task execution has failed',
      [EventType.SYSTEM_ERROR]: 'System error has occurred',
      [EventType.PERFORMANCE_ALERT]: 'Performance threshold exceeded',
      [EventType.HEALTH_CHECK]: 'System health check completed'
    };
    return descriptions[type] || 'System event occurred';
  }

  // ===== EVENT SUBSCRIPTIONS =====

  public onNewEvent(listener: EventListener<SystemEvent>): Subscription {
    return this.subscribe('newEvent', listener);
  }

  public onCriticalEvent(listener: EventListener<SystemEvent>): Subscription {
    return this.subscribe('criticalEvent', listener);
  }

  public onErrorEvent(listener: EventListener<SystemEvent>): Subscription {
    return this.subscribe('errorEvent', listener);
  }

  public onEventsLoaded(listener: EventListener<EventTimeline>): Subscription {
    return this.subscribe('eventsLoaded', listener);
  }

  public onEventAcknowledged(listener: EventListener<SystemEvent>): Subscription {
    return this.subscribe('eventAcknowledged', listener);
  }

  public onWebSocketConnected(listener: EventListener<void>): Subscription {
    return this.subscribe('websocketConnected', listener);
  }

  public onWebSocketDisconnected(listener: EventListener<void>): Subscription {
    return this.subscribe('websocketDisconnected', listener);
  }

  public onRealtimeMonitoringStarted(listener: EventListener<void>): Subscription {
    return this.subscribe('realtimeMonitoringStarted', listener);
  }

  public onRealtimeMonitoringStopped(listener: EventListener<void>): Subscription {
    return this.subscribe('realtimeMonitoringStopped', listener);
  }

  // ===== UTILITY METHODS =====

  /**
   * Get event timeline (local cache)
   */
  getEventTimeline(): SystemEvent[] {
    return [...this.eventTimeline];
  }

  /**
   * Clear acknowledged events from local cache
   */
  clearAcknowledgedEvents(): void {
    for (const [id, event] of this.systemEvents) {
      if (event.acknowledged) {
        this.systemEvents.delete(id);
      }
    }
    
    this.eventTimeline = this.eventTimeline.filter(event => !event.acknowledged);
    this.emit('acknowledgedEventsCleared');
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopRealtimeMonitoring();
    this.systemEvents.clear();
    this.eventTimeline = [];
    this.connectionRetries = 0;
    super.destroy();
  }
}

// Singleton instance
let eventService: EventService | null = null;

export function getEventService(config?: Partial<ServiceConfig>): EventService {
  if (!eventService) {
    eventService = new EventService(config);
  }
  return eventService;
}

export function resetEventService(): void {
  if (eventService) {
    eventService.destroy();
    eventService = null;
  }
}
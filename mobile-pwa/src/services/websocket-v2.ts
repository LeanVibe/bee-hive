/**
 * WebSocket v2 Client - Real-time connection to API v2 WebSocket endpoint
 * 
 * Connects directly to the API v2 WebSocket endpoint for real-time agent
 * and task updates from the SimpleOrchestrator.
 */

import { EventEmitter } from '../utils/event-emitter';

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
  agent_id?: string;
  task_id?: string;
}

export class WebSocketV2Client extends EventEmitter {
  private ws: WebSocket | null = null;
  private clientId: string;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private reconnectInterval: number = 5000;
  private isConnecting: boolean = false;
  private subscriptions: Set<string> = new Set();

  constructor() {
    super();
    this.clientId = `pwa-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Connect to the API v2 WebSocket endpoint
   */
  async connect(): Promise<void> {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      console.log('🔌 WebSocket v2 already connected or connecting');
      return;
    }

    this.isConnecting = true;
    const wsUrl = `ws://localhost:8000/api/v2/ws/${this.clientId}`;
    
    console.log('🚀 Connecting to WebSocket v2:', wsUrl);

    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('✅ WebSocket v2 connected successfully');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        
        // Subscribe to all agents and tasks by default
        this.subscribeToAgents();
        this.subscribeToTasks();
        
        this.emit('connected', { clientId: this.clientId });
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('❌ Failed to parse WebSocket v2 message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('🔌 WebSocket v2 connection closed:', event.code, event.reason);
        this.isConnecting = false;
        this.ws = null;
        this.emit('disconnected', { code: event.code, reason: event.reason });
        
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket v2 error:', error);
        this.isConnecting = false;
        this.emit('error', error);
      };

    } catch (error) {
      console.error('❌ Failed to create WebSocket v2 connection:', error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    if (this.ws) {
      console.log('🛑 Disconnecting WebSocket v2');
      this.ws.close(1000, 'User initiated disconnect');
      this.ws = null;
    }
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
  }

  /**
   * Send a message through the WebSocket
   */
  private send(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('⚠️ WebSocket v2 not connected, cannot send message:', message);
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(message: WebSocketMessage): void {
    console.log('📨 WebSocket v2 message received:', message.type);

    switch (message.type) {
      case 'connection_established':
        console.log('🎉 WebSocket v2 connection established:', message.data);
        this.emit('connectionEstablished', message.data);
        break;

      case 'agent_update':
        console.log('🤖 Agent update received:', message.agent_id, message.data);
        this.emit('agentUpdate', {
          agentId: message.agent_id,
          data: message.data,
          timestamp: message.timestamp
        });
        break;

      case 'task_update':
        console.log('📋 Task update received:', message.task_id, message.data);
        this.emit('taskUpdate', {
          taskId: message.task_id,
          data: message.data,
          timestamp: message.timestamp
        });
        break;

      case 'system_status':
        console.log('📊 System status update received:', message.data);
        this.emit('systemStatusUpdate', message.data);
        break;

      case 'agents_list':
        console.log('👥 Agents list received:', message.data?.total || 0, 'agents');
        this.emit('agentsListUpdate', message.data);
        break;

      case 'tasks_list':
        console.log('📋 Tasks list received:', message.data?.total || 0, 'tasks');
        this.emit('tasksListUpdate', message.data);
        break;

      case 'subscription_confirmed':
        console.log('✅ Subscription confirmed:', message);
        break;

      case 'error':
        console.error('❌ WebSocket v2 server error:', message);
        this.emit('serverError', message);
        break;

      default:
        console.log('📦 Unknown message type:', message.type);
    }
  }

  /**
   * Subscribe to agent updates
   */
  subscribeToAgents(agentId: string = '*'): void {
    console.log('🔔 Subscribing to agents:', agentId);
    this.subscriptions.add(`agent:${agentId}`);
    this.send({
      command: 'subscribe_agent',
      agent_id: agentId
    });
  }

  /**
   * Subscribe to task updates
   */
  subscribeToTasks(taskId: string = '*'): void {
    console.log('🔔 Subscribing to tasks:', taskId);
    this.subscriptions.add(`task:${taskId}`);
    this.send({
      command: 'subscribe_task',
      task_id: taskId
    });
  }

  /**
   * Unsubscribe from agent updates
   */
  unsubscribeFromAgents(agentId: string = '*'): void {
    console.log('🔕 Unsubscribing from agents:', agentId);
    this.subscriptions.delete(`agent:${agentId}`);
    this.send({
      command: 'unsubscribe_agent',
      agent_id: agentId
    });
  }

  /**
   * Unsubscribe from task updates
   */
  unsubscribeFromTasks(taskId: string = '*'): void {
    console.log('🔕 Unsubscribing from tasks:', taskId);
    this.subscriptions.delete(`task:${taskId}`);
    this.send({
      command: 'unsubscribe_task',
      task_id: taskId
    });
  }

  /**
   * Request current system status
   */
  requestSystemStatus(): void {
    console.log('📊 Requesting system status');
    this.send({
      command: 'get_system_status'
    });
  }

  /**
   * Request current agents list
   */
  requestAgentsList(): void {
    console.log('👥 Requesting agents list');
    this.send({
      command: 'list_agents'
    });
  }

  /**
   * Request current tasks list
   */
  requestTasksList(): void {
    console.log('📋 Requesting tasks list');
    this.send({
      command: 'list_tasks'
    });
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn('⚠️ Max WebSocket v2 reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(2, Math.min(this.reconnectAttempts - 1, 3));
    
    console.log(`🔄 Scheduling WebSocket v2 reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
        this.connect().catch(error => {
          console.error('❌ WebSocket v2 reconnection failed:', error);
        });
      }
    }, delay);
  }

  /**
   * Get connection status
   */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get client ID
   */
  get getClientId(): string {
    return this.clientId;
  }
}

// Export singleton instance
export const webSocketV2Client = new WebSocketV2Client();
export default webSocketV2Client;
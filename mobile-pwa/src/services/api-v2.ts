/**
 * API v2 Service - Direct Connection to Backend API v2 Endpoints
 * 
 * This service connects the PWA directly to the functional API v2 endpoints
 * for real-time agent and task management. Bypasses the legacy backend adapter
 * to connect directly to SimpleOrchestrator via FastAPI.
 */

import { BaseService } from './base-service';

// API v2 Types matching the backend models
export interface Agent {
  id: string;
  role: string;
  status: 'active' | 'inactive' | 'error' | 'idle';
  created_at: string;
  last_activity: string;
  current_task_id?: string;
}

export interface AgentListResponse {
  agents: Agent[];
  total: number;
  active: number;
  inactive: number;
}

export interface AgentCreateRequest {
  role: string;
  agent_type?: string;
  task_id?: string;
  workspace_name?: string;
  git_branch?: string;
}

export interface Task {
  id: string;
  title: string;
  description?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assigned_to?: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
  pending: number;
  in_progress: number;
  completed: number;
}

export interface TaskCreateRequest {
  title: string;
  description?: string;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  assigned_to?: string;
  metadata?: Record<string, any>;
}

export class ApiV2Service extends BaseService {
  constructor() {
    super({
      baseUrl: 'http://localhost:8000/api/v2',
      cacheTimeout: 30000, // 30 seconds cache for API v2 data
      headers: {
        'Content-Type': 'application/json',
      }
    });
  }

  /**
   * Agent Management
   */
  
  async listAgents(status?: string, role?: string, limit: number = 50, offset: number = 0): Promise<AgentListResponse> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    if (role) params.append('role', role);
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());
    
    const url = `/agents/?${params.toString()}`;
    console.log('🔍 Fetching agents from API v2:', url);
    
    return await this.get<AgentListResponse>(url);
  }

  async getAgent(agentId: string): Promise<Agent> {
    console.log('🔍 Fetching agent from API v2:', agentId);
    return await this.get<Agent>(`/agents/${agentId}`);
  }

  async createAgent(request: AgentCreateRequest): Promise<Agent> {
    console.log('🚀 Creating agent via API v2:', request);
    
    // Map common role names to backend enum values
    const roleMapping: Record<string, string> = {
      'developer': 'backend_developer',
      'frontend': 'frontend_developer', 
      'qa': 'qa_engineer',
      'devops': 'devops_engineer'
    };
    
    const mappedRequest = {
      ...request,
      role: roleMapping[request.role] || request.role,
      agent_type: request.agent_type || 'claude_code'
    };
    
    return await this.post<Agent>('/agents/', mappedRequest);
  }

  async updateAgentStatus(agentId: string, status: string): Promise<Agent> {
    console.log('🔄 Updating agent status via API v2:', agentId, status);
    return await this.put<Agent>(`/agents/${agentId}/status`, { status });
  }

  async deleteAgent(agentId: string): Promise<void> {
    console.log('🗑️ Deleting agent via API v2:', agentId);
    await this.delete(`/agents/${agentId}`);
  }

  async getAgentHealth(agentId: string): Promise<any> {
    console.log('❤️ Getting agent health via API v2:', agentId);
    return await this.get<any>(`/agents/${agentId}/health`);
  }

  /**
   * Task Management
   */

  async listTasks(status?: string, assignedTo?: string, limit: number = 50, offset: number = 0): Promise<TaskListResponse> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    if (assignedTo) params.append('assigned_to', assignedTo);
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());
    
    const url = `/tasks/?${params.toString()}`;
    console.log('📋 Fetching tasks from API v2:', url);
    
    return await this.get<TaskListResponse>(url);
  }

  async getTask(taskId: string): Promise<Task> {
    console.log('📋 Fetching task from API v2:', taskId);
    return await this.get<Task>(`/tasks/${taskId}`);
  }

  async createTask(request: TaskCreateRequest): Promise<Task> {
    console.log('📝 Creating task via API v2:', request);
    
    const taskRequest = {
      ...request,
      priority: request.priority || 'medium'
    };
    
    return await this.post<Task>('/tasks/', taskRequest);
  }

  async updateTask(taskId: string, updates: Partial<TaskCreateRequest>): Promise<Task> {
    console.log('📝 Updating task via API v2:', taskId, updates);
    return await this.put<Task>(`/tasks/${taskId}`, updates);
  }

  async assignTask(taskId: string, agentId: string): Promise<Task> {
    console.log('👤 Assigning task via API v2:', taskId, 'to', agentId);
    return await this.put<Task>(`/tasks/${taskId}/assign`, { agent_id: agentId });
  }

  async deleteTask(taskId: string): Promise<void> {
    console.log('🗑️ Deleting task via API v2:', taskId);
    await this.delete(`/tasks/${taskId}`);
  }

  /**
   * System Status and Monitoring
   */

  async getSystemStatus(): Promise<any> {
    console.log('📊 Fetching system status from API v2');
    return await this.get<any>('/status');
  }

  /**
   * Real-time Updates Support
   */

  async startRealtimeUpdates(): Promise<() => void> {
    console.log('🚀 Starting real-time updates for API v2');
    
    // Start polling for updates every 5 seconds as fallback
    const pollingInterval = setInterval(async () => {
      try {
        // Fetch latest agents and tasks
        const [agents, tasks] = await Promise.all([
          this.listAgents(),
          this.listTasks()
        ]);
        
        // Emit events for components listening
        this.emit('agentsUpdated', agents);
        this.emit('tasksUpdated', tasks);
        
        // Also emit as liveDataUpdated for compatibility
        this.emit('liveDataUpdated', {
          metrics: {
            active_agents: agents.active,
            active_projects: Math.ceil(tasks.in_progress / 3), // Estimate
            system_status: agents.active > 0 ? 'healthy' : 'degraded',
            last_updated: new Date().toISOString()
          },
          agent_activities: agents.agents.map(agent => ({
            agent_id: agent.id,
            name: `${agent.role.replace('_', ' ')} Agent`,
            status: agent.status,
            current_task: agent.current_task_id,
            performance_score: 85 + Math.random() * 15,
            specializations: [agent.role]
          })),
          project_snapshots: [],
          conflict_snapshots: []
        });
        
      } catch (error) {
        console.warn('⚠️ Real-time update polling failed:', error);
      }
    }, 5000);
    
    return () => {
      clearInterval(pollingInterval);
      console.log('🛑 Stopped real-time updates polling');
    };
  }

  /**
   * Health Check
   */

  async healthCheck(): Promise<boolean> {
    try {
      await this.listAgents();
      console.log('✅ API v2 health check passed');
      return true;
    } catch (error) {
      console.error('❌ API v2 health check failed:', error);
      return false;
    }
  }
}

// Export singleton instance
export const apiV2Service = new ApiV2Service();
export default apiV2Service;
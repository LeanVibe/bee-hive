/**
 * Backend Adapter Service
 * 
 * Maps PWA service layer to working LeanVibe backend endpoints
 * Uses /dashboard/api/live-data as primary data source and transforms
 * data into the format expected by PWA services
 */

import { BaseService } from './base-service';

export interface LiveDashboardData {
  metrics: {
    active_projects: number;
    active_agents: number;
    agent_utilization: number;
    completed_tasks: number;
    active_conflicts: number;
    system_efficiency: number;
    system_status: 'healthy' | 'degraded' | 'critical';
    last_updated: string;
  };
  agent_activities: Array<{
    agent_id: string;
    name: string;
    status: 'active' | 'busy' | 'idle' | 'error';
    current_project?: string;
    current_task?: string;
    task_progress?: number;
    performance_score: number;
    specializations: string[];
  }>;
  project_snapshots: Array<{
    name: string;
    status: 'active' | 'planning' | 'completed';
    progress_percentage: number;
    participating_agents: string[];
    completed_tasks: number;
    active_tasks: number;
    conflicts: number;
    quality_score: number;
  }>;
  conflict_snapshots: Array<{
    conflict_type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    project_name: string;
    description: string;
    affected_agents: string[];
    impact_score: number;
    auto_resolvable: boolean;
  }>;
}

export class BackendAdapter extends BaseService {
  private liveData: LiveDashboardData | null = null;
  private lastFetch: number = 0;
  private fetchInterval: number = 5000; // 5 seconds
  
  constructor() {
    super({
      baseUrl: '', // Use empty baseUrl so requests go through Vite proxy
      cacheTimeout: 5000 // 5 second cache for live data
    });
  }

  /**
   * Get fresh data from the working backend endpoint
   */
  async getLiveData(forceRefresh = false): Promise<LiveDashboardData> {
    const now = Date.now();
    
    // Return cached data if recent and not forced refresh
    if (!forceRefresh && this.liveData && (now - this.lastFetch) < this.fetchInterval) {
      return this.liveData;
    }

    try {
      console.log('🔄 Fetching live data from backend...');
      this.liveData = await this.get<LiveDashboardData>('/dashboard/api/live-data');
      this.lastFetch = now;
      
      // Emit update event for real-time components
      this.emit('liveDataUpdated', this.liveData);
      
      return this.liveData;
    } catch (error) {
      console.error('❌ Failed to fetch live data:', error);
      
      // Return cached data if available, otherwise throw
      if (this.liveData) {
        console.warn('⚠️ Using cached data due to fetch error');
        return this.liveData;
      }
      throw error;
    }
  }

  /**
   * Transform live data into Task format for PWA services
   */
  async getTasksFromLiveData(): Promise<any[]> {
    const data = await this.getLiveData();
    const tasks: any[] = [];
    
    // Convert agent activities to tasks
    data.agent_activities.forEach((agent, index) => {
      if (agent.current_task) {
        tasks.push({
          id: `task-${agent.agent_id}-${index}`,
          title: agent.current_task,
          description: `Task assigned to ${agent.name}`,
          status: agent.status === 'active' || agent.status === 'busy' ? 'in-progress' : 'pending',
          priority: 'medium',
          assignedTo: agent.agent_id,
          assignedToName: agent.name,
          progress: agent.task_progress || 0,
          tags: agent.specializations,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          projectId: agent.current_project || 'default-project'
        });
      }
    });

    // Add project-based tasks
    data.project_snapshots.forEach((project, index) => {
      for (let i = 0; i < project.active_tasks; i++) {
        tasks.push({
          id: `project-task-${index}-${i}`,
          title: `${project.name} - Active Task ${i + 1}`,
          description: `Active task in ${project.name} project`,
          status: 'in-progress',
          priority: 'high',
          progress: project.progress_percentage,
          tags: ['project', project.status],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          projectId: project.name.toLowerCase().replace(/\s+/g, '-')
        });
      }
      
      for (let i = 0; i < project.completed_tasks; i++) {
        tasks.push({
          id: `project-completed-${index}-${i}`,
          title: `${project.name} - Completed Task ${i + 1}`,
          description: `Completed task in ${project.name} project`,
          status: 'done',
          priority: 'medium',
          progress: 100,
          tags: ['project', 'completed'],
          createdAt: new Date(Date.now() - (i + 1) * 3600000).toISOString(), // Stagger completion times
          updatedAt: new Date(Date.now() - (i + 1) * 3600000).toISOString(),
          projectId: project.name.toLowerCase().replace(/\s+/g, '-')
        });
      }
    });

    return tasks;
  }

  /**
   * Transform live data into Agent format for PWA services
   */
  async getAgentsFromLiveData(): Promise<any[]> {
    const data = await this.getLiveData();
    
    return data.agent_activities.map(agent => ({
      id: agent.agent_id,
      name: agent.name,
      status: agent.status,
      role: agent.name.toLowerCase().replace(/\s+/g, '_'),
      capabilities: agent.specializations,
      current_task_id: agent.current_task ? `task-${agent.agent_id}` : null,
      current_project: agent.current_project,
      performance_score: agent.performance_score,
      uptime: Math.floor(Math.random() * 86400), // Simulated uptime in seconds
      last_seen: new Date().toISOString(),
      performance_metrics: {
        cpu_usage: [Math.random() * 100],
        memory_usage: [Math.random() * 100],
        token_usage: [Math.floor(Math.random() * 10000)],
        tasks_completed: [Math.floor(Math.random() * 50)],
        error_rate: [Math.random() * 5],
        response_time: [Math.random() * 2000],
        timestamps: [new Date().toISOString()],
        overall_score: agent.performance_score,
        trend: 'stable' as const
      }
    }));
  }

  /**
   * Transform live data into System Health format
   */
  async getSystemHealthFromLiveData() {
    const data = await this.getLiveData();
    
    return {
      overall: data.metrics.system_status,
      components: {
        healthy: data.agent_activities.filter(a => a.status === 'active').length,
        degraded: data.agent_activities.filter(a => a.status === 'busy').length,
        unhealthy: data.agent_activities.filter(a => a.status === 'error').length
      },
      last_updated: data.metrics.last_updated
    };
  }

  /**
   * Transform live data into Events format
   */
  async getEventsFromLiveData(limit = 50) {
    const data = await this.getLiveData();
    const events: any[] = [];
    
    // Generate events from agent activities
    data.agent_activities.forEach(agent => {
      events.push({
        id: `agent-status-${agent.agent_id}`,
        event_type: 'agent_status_change',
        summary: `${agent.name} is ${agent.status}`,
        description: agent.current_task || 'No active task',
        agent_id: agent.agent_id,
        created_at: new Date().toISOString(),
        severity: agent.status === 'error' ? 'high' : 'medium',
        metadata: {
          performance_score: agent.performance_score,
          specializations: agent.specializations
        }
      });
    });

    // Generate events from conflicts
    data.conflict_snapshots.forEach(conflict => {
      events.push({
        id: `conflict-${conflict.conflict_type}`,
        event_type: 'conflict_detected',
        summary: `${conflict.conflict_type} in ${conflict.project_name}`,
        description: conflict.description,
        agent_id: conflict.affected_agents[0],
        created_at: new Date().toISOString(),
        severity: conflict.severity,
        metadata: {
          impact_score: conflict.impact_score,
          auto_resolvable: conflict.auto_resolvable
        }
      });
    });

    // Generate events from project updates
    data.project_snapshots.forEach(project => {
      events.push({
        id: `project-${project.name}`,
        event_type: 'project_progress',
        summary: `${project.name} at ${Math.round(project.progress_percentage)}% completion`,
        description: `${project.completed_tasks} tasks completed, ${project.active_tasks} in progress`,
        created_at: new Date().toISOString(),
        severity: 'info',
        metadata: {
          progress: project.progress_percentage,
          quality_score: project.quality_score
        }
      });
    });

    return events.slice(0, limit);
  }

  /**
   * Get performance metrics from live data
   */
  async getPerformanceMetricsFromLiveData() {
    const data = await this.getLiveData();
    
    return {
      system_metrics: {
        cpu_usage: data.metrics.system_efficiency,
        memory_usage: 100 - data.metrics.system_efficiency, // Inverse for demo
        network_usage: Math.random() * 100,
        disk_usage: Math.random() * 100
      },
      agent_metrics: data.agent_activities.reduce((acc, agent) => {
        acc[agent.agent_id] = {
          performance_score: agent.performance_score,
          task_completion_rate: Math.random() * 100,
          error_rate: Math.random() * 5,
          uptime: Math.random() * 100
        };
        return acc;
      }, {} as Record<string, any>),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Start real-time polling of live data
   */
  startRealtimeUpdates(): () => void {
    console.log('🚀 Starting real-time updates from backend...');
    
    return this.startPolling(async () => {
      await this.getLiveData(true); // Force refresh on polling
    }, this.fetchInterval);
  }

  /**
   * Mock write operations for services that need them
   * Since we don't have write endpoints, these return success
   */
  async mockWriteOperation(operation: string, data: any): Promise<any> {
    console.log(`🔧 Mock ${operation} operation:`, data);
    
    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Return success response with the data plus some generated fields
    return {
      ...data,
      id: data.id || `mock-${Date.now()}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      status: 'success'
    };
  }
}

// Export singleton instance
export const backendAdapter = new BackendAdapter();
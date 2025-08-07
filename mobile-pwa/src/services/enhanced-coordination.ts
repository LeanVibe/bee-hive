/**
 * Enhanced Multi-Agent Coordination Service for LeanVibe Agent Hive
 * 
 * Connects mobile dashboard to the sophisticated autonomous development capabilities
 * in enhanced_multi_agent_coordination.py, providing real-time visibility into
 * advanced multi-agent orchestration, task collaboration, and autonomous development.
 */

import { BaseService } from './base-service';

export interface SpecializedAgent {
  id: string;
  role: string;
  name: string;
  status: 'active' | 'busy' | 'idle' | 'error';
  capabilities: string[];
  current_task?: string;
  performance_score: number;
  collaboration_history: number;
}

export interface CoordinationPattern {
  id: string;
  type: string;
  name: string;
  description: string;
  agents_required: number;
  estimated_duration: number;
  success_rate: number;
  last_executed?: string;
}

export interface CollaborationSession {
  id: string;
  pattern_type: string;
  agents: SpecializedAgent[];
  status: 'planning' | 'executing' | 'review' | 'completed';
  progress: number;
  started_at: string;
  estimated_completion?: string;
  tasks: CollaborationTask[];
}

export interface CollaborationTask {
  id: string;
  title: string;
  description: string;
  assigned_to: string[];
  status: 'pending' | 'in_progress' | 'review' | 'completed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  dependencies: string[];
  progress: number;
}

export interface EnhancedCoordinationStatus {
  total_agents: number;
  available_agents: number;
  active_collaborations: number;
  completed_tasks_today: number;
  success_rate: number;
  average_task_completion_time: number;
  autonomous_development_progress: {
    features_developed: number;
    code_reviews_completed: number;
    tests_written: number;
    deployments_automated: number;
  };
}

export interface DemonstrationResult {
  demonstration_id: string;
  status: 'executing' | 'completed' | 'failed';
  patterns_demonstrated: number;
  success_count: number;
  failure_count: number;
  insights: string[];
  business_value_metrics: {
    time_saved_minutes: number;
    quality_improvement_percent: number;
    developer_productivity_multiplier: number;
  };
}

export class EnhancedCoordinationService extends BaseService {
  private wsConnection: WebSocket | null = null;

  constructor(config: { apiUrl: string; wsUrl?: string }) {
    super(config);
  }

  /**
   * Get current enhanced coordination system status
   */
  async getCoordinationStatus(): Promise<EnhancedCoordinationStatus> {
    return this.get<EnhancedCoordinationStatus>('/api/v1/enhanced-coordination/status');
  }

  /**
   * Get all specialized agents in the enhanced system
   */
  async getSpecializedAgents(): Promise<SpecializedAgent[]> {
    const response = await this.get<{ agents: SpecializedAgent[] }>('/api/v1/enhanced-coordination/agents');
    return response.agents;
  }

  /**
   * Get available coordination patterns
   */
  async getCoordinationPatterns(): Promise<CoordinationPattern[]> {
    const response = await this.get<{ patterns: CoordinationPattern[] }>('/api/v1/enhanced-coordination/patterns');
    return response.patterns;
  }

  /**
   * Get active collaboration sessions
   */
  async getActiveCollaborations(): Promise<CollaborationSession[]> {
    const response = await this.get<{ collaborations: CollaborationSession[] }>('/api/v1/enhanced-coordination/collaborate');
    return response.collaborations;
  }

  /**
   * Start autonomous development demonstration
   */
  async startDemonstration(): Promise<DemonstrationResult> {
    return this.post<DemonstrationResult>('/api/v1/enhanced-coordination/demonstration');
  }

  /**
   * Get coordination analytics for business value assessment
   */
  async getCoordinationAnalytics(timeRange: '1h' | '24h' | '7d' | '30d' = '24h'): Promise<{
    autonomous_development_metrics: {
      features_completed: number;
      code_quality_score: number;
      test_coverage_improvement: number;
      deployment_frequency: number;
    };
    developer_empowerment_metrics: {
      decisions_automated: number;
      human_intervention_rate: number;
      time_to_decision_seconds: number;
      confidence_score: number;
    };
    business_impact_metrics: {
      productivity_gain_percent: number;
      quality_improvement_percent: number;
      time_saved_hours: number;
      cost_reduction_percent: number;
    };
  }> {
    return this.get(`/api/v1/enhanced-coordination/analytics?timeRange=${timeRange}`);
  }

  /**
   * Form specialized development team
   */
  async formDevelopmentTeam(requirements: {
    project_type: string;
    complexity: 'simple' | 'moderate' | 'complex';
    timeline: string;
    skills_required: string[];
  }): Promise<{
    team_id: string;
    agents: SpecializedAgent[];
    coordination_pattern: string;
    estimated_completion: string;
  }> {
    return this.post('/api/v1/enhanced-coordination/teams/form', requirements);
  }

  /**
   * Execute coordination pattern with real-time updates
   */
  async executePattern(patternId: string, parameters: Record<string, any>): Promise<{
    execution_id: string;
    status: string;
    estimated_duration: number;
  }> {
    return this.post('/api/v1/enhanced-coordination/patterns/execute', {
      pattern_id: patternId,
      parameters
    });
  }

  /**
   * Connect to real-time coordination updates via WebSocket
   */
  connectToCoordinationUpdates(callbacks: {
    onAgentStatusChange?: (agent: SpecializedAgent) => void;
    onCollaborationUpdate?: (session: CollaborationSession) => void;
    onTaskComplete?: (task: CollaborationTask) => void;
    onBusinessMetricsUpdate?: (metrics: any) => void;
  }): void {
    if (this.wsConnection) {
      this.wsConnection.close();
    }

    const wsUrl = this.config.wsUrl || this.config.apiUrl.replace('http', 'ws') + '/ws/enhanced-coordination';
    this.wsConnection = new WebSocket(wsUrl);

    this.wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'agent_status_change':
            callbacks.onAgentStatusChange?.(data.agent);
            break;
          case 'collaboration_update':
            callbacks.onCollaborationUpdate?.(data.session);
            break;
          case 'task_complete':
            callbacks.onTaskComplete?.(data.task);
            break;
          case 'business_metrics_update':
            callbacks.onBusinessMetricsUpdate?.(data.metrics);
            break;
        }
      } catch (error) {
        console.error('Failed to parse coordination WebSocket message:', error);
      }
    };

    this.wsConnection.onerror = (error) => {
      console.error('Enhanced coordination WebSocket error:', error);
    };

    this.wsConnection.onclose = () => {
      console.log('Enhanced coordination WebSocket closed');
      // Auto-reconnect after 5 seconds
      setTimeout(() => this.connectToCoordinationUpdates(callbacks), 5000);
    };
  }

  /**
   * Disconnect from real-time updates
   */
  disconnect(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  /**
   * Get real-time decision points requiring human input
   */
  async getHumanDecisionPoints(): Promise<{
    urgent_decisions: Array<{
      id: string;
      title: string;
      description: string;
      options: string[];
      impact: 'low' | 'medium' | 'high';
      deadline: string;
      context: Record<string, any>;
    }>;
    system_recommendations: Array<{
      decision_id: string;
      recommended_action: string;
      confidence: number;
      reasoning: string;
    }>;
  }> {
    return this.get('/api/v1/enhanced-coordination/decisions/pending');
  }

  /**
   * Submit human decision to autonomous system
   */
  async submitDecision(decisionId: string, choice: string, feedback?: string): Promise<{
    success: boolean;
    impact_assessment: {
      affected_agents: string[];
      timeline_change: string;
      confidence_adjustment: number;
    };
  }> {
    return this.post(`/api/v1/enhanced-coordination/decisions/${decisionId}/submit`, {
      choice,
      feedback
    });
  }
}
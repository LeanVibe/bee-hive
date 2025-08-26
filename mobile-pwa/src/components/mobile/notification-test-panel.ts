/**
 * Phase 4 Notification Test Panel
 * 
 * Testing component for demonstrating Phase 4 notification capabilities
 * including agent events, system alerts, and mobile optimizations.
 */

import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { getPhase4NotificationHandler, notifyAgentError, notifyBuildFailure, notifyCriticalSystemIssue, notifyTaskCompletion } from '../../services/phase4-notification-handler';

@customElement('notification-test-panel')
export class NotificationTestPanel extends LitElement {
  @state() permissionStatus: NotificationPermission = 'default';
  @state() isSubscribed = false;
  @state() testResults: string[] = [];
  @state() deliveryStats = { sent: 0, failed: 0, queued: 0, avgDeliveryTime: 0 };

  private notificationHandler = getPhase4NotificationHandler();

  static styles = css`
    :host {
      display: block;
      padding: 1rem;
      background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
      border-radius: 12px;
      margin: 1rem 0;
      border: 1px solid #475569;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid #475569;
    }

    .panel-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #e2e8f0;
      margin: 0;
    }

    .permission-status {
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 500;
      text-transform: uppercase;
    }

    .permission-granted {
      background: #10b981;
      color: white;
    }

    .permission-denied {
      background: #ef4444;
      color: white;
    }

    .permission-default {
      background: #f59e0b;
      color: white;
    }

    .test-section {
      margin: 1rem 0;
    }

    .section-title {
      font-size: 0.9rem;
      font-weight: 500;
      color: #cbd5e1;
      margin-bottom: 0.5rem;
    }

    .test-buttons {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .test-btn {
      padding: 0.75rem 1rem;
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }

    .test-btn:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }

    .test-btn:active {
      transform: translateY(0);
    }

    .test-btn.critical {
      background: #ef4444;
    }

    .test-btn.critical:hover {
      background: #dc2626;
    }

    .test-btn.warning {
      background: #f59e0b;
    }

    .test-btn.warning:hover {
      background: #d97706;
    }

    .test-btn.success {
      background: #10b981;
    }

    .test-btn.success:hover {
      background: #059669;
    }

    .setup-buttons {
      display: flex;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .setup-btn {
      padding: 0.5rem 1rem;
      background: #6366f1;
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .setup-btn:hover {
      background: #5b21b6;
    }

    .setup-btn:disabled {
      background: #64748b;
      cursor: not-allowed;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 0.75rem;
      margin-top: 1rem;
    }

    .stat-card {
      background: #334155;
      padding: 0.75rem;
      border-radius: 8px;
      text-align: center;
    }

    .stat-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #3b82f6;
      margin-bottom: 0.25rem;
    }

    .stat-label {
      font-size: 0.75rem;
      color: #94a3b8;
      text-transform: uppercase;
    }

    .test-results {
      background: #0f172a;
      padding: 1rem;
      border-radius: 8px;
      margin-top: 1rem;
      max-height: 200px;
      overflow-y: auto;
    }

    .result-item {
      font-size: 0.8rem;
      color: #94a3b8;
      margin-bottom: 0.5rem;
      padding: 0.25rem 0;
      border-bottom: 1px solid #1e293b;
    }

    .result-item:last-child {
      border-bottom: none;
    }

    .result-timestamp {
      color: #64748b;
      font-size: 0.7rem;
    }

    @media (max-width: 768px) {
      .test-buttons {
        grid-template-columns: 1fr;
      }

      .setup-buttons {
        flex-direction: column;
      }

      .stats-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }
  `;

  async connectedCallback() {
    super.connectedCallback();
    await this.updateStatus();
    this.setupEventListeners();
  }

  private async updateStatus() {
    this.permissionStatus = Notification.permission;
    this.isSubscribed = this.notificationHandler.getNotificationStats().isSubscribed;
    this.deliveryStats = this.notificationHandler.getDeliveryStats();
  }

  private setupEventListeners() {
    this.notificationHandler.on('agent_event_processed', () => {
      this.addTestResult('✅ Agent event notification sent');
      this.updateStatus();
    });

    this.notificationHandler.on('system_alert_processed', () => {
      this.addTestResult('🚨 System alert notification sent');
      this.updateStatus();
    });

    this.notificationHandler.on('agent_event_failed', (data) => {
      this.addTestResult(`❌ Agent event failed: ${data.error.message}`);
      this.updateStatus();
    });
  }

  private addTestResult(message: string) {
    const timestamp = new Date().toLocaleTimeString();
    this.testResults = [
      `[${timestamp}] ${message}`,
      ...this.testResults.slice(0, 19) // Keep last 20 results
    ];
  }

  private async requestPermissions() {
    try {
      const granted = await this.notificationHandler.requestPermissions();
      this.addTestResult(`Permission ${granted ? 'granted' : 'denied'}`);
      await this.updateStatus();
    } catch (error) {
      this.addTestResult(`Permission request failed: ${error.message}`);
    }
  }

  private async subscribeToPush() {
    try {
      const subscribed = await this.notificationHandler.subscribeToPushNotifications();
      this.addTestResult(`Push subscription ${subscribed ? 'successful' : 'failed'}`);
      await this.updateStatus();
    } catch (error) {
      this.addTestResult(`Push subscription failed: ${error.message}`);
    }
  }

  private async testAgentError() {
    try {
      await notifyAgentError(
        'test-agent-001',
        'Test Agent Alpha',
        'Simulated critical agent error for testing Phase 4 notifications',
        { 
          testMode: true,
          timestamp: Date.now(),
          severity: 'critical'
        }
      );
      this.addTestResult('🔥 Agent error notification triggered');
    } catch (error) {
      this.addTestResult(`Agent error test failed: ${error.message}`);
    }
  }

  private async testBuildFailure() {
    try {
      await notifyBuildFailure(
        'build-12345',
        'Mobile PWA Build Service',
        'Test build failure: TypeScript compilation error in Phase 4 components',
        { 
          testMode: true,
          buildNumber: 12345,
          errorCode: 'TS2345'
        }
      );
      this.addTestResult('🔨 Build failure notification triggered');
    } catch (error) {
      this.addTestResult(`Build failure test failed: ${error.message}`);
    }
  }

  private async testCriticalSystem() {
    try {
      await notifyCriticalSystemIssue(
        'Database Service',
        'Test critical system issue: Database connection pool exhausted',
        'major',
        { 
          testMode: true,
          connectionPool: 'exhausted',
          impact: 'all-services'
        }
      );
      this.addTestResult('💥 Critical system notification triggered');
    } catch (error) {
      this.addTestResult(`Critical system test failed: ${error.message}`);
    }
  }

  private async testTaskCompletion() {
    try {
      await notifyTaskCompletion(
        'test-agent-002',
        'Test Agent Beta',
        'Phase 4 notification system integration testing',
        { 
          testMode: true,
          duration: '5.2s',
          success: true
        }
      );
      this.addTestResult('✅ Task completion notification triggered');
    } catch (error) {
      this.addTestResult(`Task completion test failed: ${error.message}`);
    }
  }

  private async testMobileOptimized() {
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    try {
      await this.notificationHandler.handleAgentEvent({
        id: crypto.randomUUID(),
        agentId: 'mobile-test-agent',
        agentName: 'Mobile Test Agent',
        type: 'status_change',
        severity: 'medium',
        title: 'Mobile Notification Test',
        message: `Mobile-optimized notification test. Device: ${isMobile ? 'Mobile' : 'Desktop'}. Touch support: ${'ontouchstart' in window}`,
        metadata: {
          testMode: true,
          isMobile,
          touchSupport: 'ontouchstart' in window,
          viewportWidth: window.innerWidth
        },
        timestamp: Date.now(),
        actionRequired: false
      });
      this.addTestResult(`📱 Mobile optimized test (${isMobile ? 'Mobile' : 'Desktop'} detected)`);
    } catch (error) {
      this.addTestResult(`Mobile optimized test failed: ${error.message}`);
    }
  }

  private clearResults() {
    this.testResults = [];
    this.addTestResult('Test results cleared');
  }

  render() {
    return html`
      <div class="panel-header">
        <h3 class="panel-title">🔔 Phase 4 Notification Test Panel</h3>
        <div class="permission-status ${this.permissionStatus === 'granted' ? 'permission-granted' : 
                                       this.permissionStatus === 'denied' ? 'permission-denied' : 'permission-default'}">
          ${this.permissionStatus}
        </div>
      </div>

      <div class="test-section">
        <div class="section-title">Setup & Permissions</div>
        <div class="setup-buttons">
          <button class="setup-btn" @click=${this.requestPermissions} 
                  ?disabled=${this.permissionStatus === 'granted'}>
            Request Permission
          </button>
          <button class="setup-btn" @click=${this.subscribeToPush} 
                  ?disabled=${this.isSubscribed || this.permissionStatus !== 'granted'}>
            Subscribe to Push
          </button>
          <button class="setup-btn" @click=${this.clearResults}>
            Clear Results
          </button>
        </div>
      </div>

      <div class="test-section">
        <div class="section-title">Notification Tests</div>
        <div class="test-buttons">
          <button class="test-btn critical" @click=${this.testAgentError}>
            🚫 Test Agent Error
          </button>
          <button class="test-btn critical" @click=${this.testBuildFailure}>
            🔥 Test Build Failure
          </button>
          <button class="test-btn critical" @click=${this.testCriticalSystem}>
            💥 Test Critical System
          </button>
          <button class="test-btn success" @click=${this.testTaskCompletion}>
            ✅ Test Task Complete
          </button>
          <button class="test-btn warning" @click=${this.testMobileOptimized}>
            📱 Test Mobile Optimized
          </button>
        </div>
      </div>

      <div class="test-section">
        <div class="section-title">Delivery Statistics</div>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">${this.deliveryStats.sent}</div>
            <div class="stat-label">Sent</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${this.deliveryStats.failed}</div>
            <div class="stat-label">Failed</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${this.deliveryStats.queued}</div>
            <div class="stat-label">Queued</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${this.deliveryStats.avgDeliveryTime.toFixed(1)}ms</div>
            <div class="stat-label">Avg Time</div>
          </div>
        </div>
      </div>

      ${this.testResults.length > 0 ? html`
        <div class="test-section">
          <div class="section-title">Test Results</div>
          <div class="test-results">
            ${this.testResults.map(result => html`
              <div class="result-item">${result}</div>
            `)}
          </div>
        </div>
      ` : ''}
    `;
  }
}
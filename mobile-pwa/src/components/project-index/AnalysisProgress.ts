/**
 * Analysis Progress Component
 * 
 * Real-time analysis monitoring with progress visualization,
 * performance metrics, and error reporting.
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ProjectIndexStore } from '../../services/project-index-store';
import type {
  AnalysisSession,
  AnalysisProgress as ProgressData,
  AnalysisPhase,
  AnalysisError,
  ComponentLoadingState,
  Subscription
} from '../../types/project-index';

@customElement('analysis-progress')
export class AnalysisProgress extends LitElement {
  @property({ type: String }) declare projectId: string;
  @property({ type: Object }) declare session?: AnalysisSession;
  @property({ type: Boolean }) declare compact: boolean;

  @state() private declare currentProgress?: ProgressData;
  @state() private declare loadingState: ComponentLoadingState;
  @state() private declare showDetails: boolean;
  @state() private declare showErrors: boolean;

  private store: ProjectIndexStore;
  private subscriptions: Subscription[] = [];

  constructor() {
    super();
    
    // Initialize properties
    this.projectId = '';
    this.compact = false;
    this.loadingState = { isLoading: false };
    this.showDetails = false;
    this.showErrors = false;
    
    this.store = ProjectIndexStore.getInstance();
    this.setupStoreSubscriptions();
  }

  static styles = css`
    :host {
      display: block;
      background: var(--surface-color, #ffffff);
      border-radius: 0.5rem;
      border: 1px solid var(--border-color, #e5e7eb);
      overflow: hidden;
    }

    .progress-container {
      padding: 1.5rem;
    }

    .progress-container.compact {
      padding: 1rem;
    }

    .progress-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }

    .progress-title {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-primary-color, #1f2937);
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .progress-controls {
      display: flex;
      gap: 0.5rem;
    }

    .control-button {
      background: none;
      border: 1px solid var(--border-color, #e5e7eb);
      color: var(--text-secondary-color, #6b7280);
      padding: 0.375rem;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
    }

    .control-button:hover {
      background: var(--hover-color, #f3f4f6);
    }

    .control-button.danger {
      color: var(--error-color, #ef4444);
      border-color: var(--error-color, #ef4444);
    }

    .control-button.danger:hover {
      background: rgba(239, 68, 68, 0.1);
    }

    .progress-status {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .status-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }

    .status-indicator.running {
      background: var(--primary-color, #3b82f6);
    }

    .status-indicator.completed {
      background: var(--success-color, #10b981);
      animation: none;
    }

    .status-indicator.failed {
      background: var(--error-color, #ef4444);
      animation: none;
    }

    .status-text {
      font-weight: 500;
      color: var(--text-primary-color, #1f2937);
    }

    .status-phase {
      color: var(--text-secondary-color, #6b7280);
      font-size: 0.875rem;
    }

    .progress-bar-container {
      margin-bottom: 1rem;
    }

    .progress-bar {
      width: 100%;
      height: 8px;
      background: var(--border-color, #e5e7eb);
      border-radius: 4px;
      overflow: hidden;
    }

    .progress-bar.compact {
      height: 6px;
    }

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--primary-color, #3b82f6), var(--primary-hover-color, #2563eb));
      transition: width 0.3s ease;
      position: relative;
    }

    .progress-fill.completed {
      background: var(--success-color, #10b981);
    }

    .progress-fill.failed {
      background: var(--error-color, #ef4444);
    }

    .progress-fill::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
      animation: shimmer 2s infinite;
    }

    .progress-text {
      display: flex;
      justify-content: space-between;
      margin-top: 0.5rem;
      font-size: 0.875rem;
      color: var(--text-secondary-color, #6b7280);
    }

    .progress-details {
      border-top: 1px solid var(--border-color, #e5e7eb);
      padding-top: 1rem;
      margin-top: 1rem;
    }

    .details-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
    }

    .detail-item {
      text-align: center;
    }

    .detail-value {
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--text-primary-color, #1f2937);
    }

    .detail-label {
      font-size: 0.75rem;
      color: var(--text-secondary-color, #6b7280);
      margin-top: 0.25rem;
    }

    .current-file {
      background: var(--surface-secondary-color, #f8fafc);
      border-radius: 0.375rem;
      padding: 0.75rem;
      margin-top: 1rem;
    }

    .current-file-label {
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--text-secondary-color, #6b7280);
      margin-bottom: 0.25rem;
    }

    .current-file-path {
      font-family: monospace;
      font-size: 0.875rem;
      color: var(--text-primary-color, #1f2937);
      word-break: break-all;
    }

    .errors-section {
      border-top: 1px solid var(--border-color, #e5e7eb);
      padding-top: 1rem;
      margin-top: 1rem;
    }

    .errors-header {
      display: flex;
      align-items: center;
      justify-content: between;
      margin-bottom: 0.75rem;
    }

    .errors-title {
      font-weight: 600;
      color: var(--error-color, #ef4444);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .error-item {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.2);
      border-radius: 0.375rem;
      padding: 0.75rem;
      margin-bottom: 0.5rem;
    }

    .error-item:last-child {
      margin-bottom: 0;
    }

    .error-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }

    .error-type {
      font-weight: 500;
      font-size: 0.875rem;
      color: var(--error-color, #ef4444);
    }

    .error-time {
      font-size: 0.75rem;
      color: var(--text-secondary-color, #6b7280);
    }

    .error-message {
      font-size: 0.875rem;
      color: var(--text-primary-color, #1f2937);
      margin-bottom: 0.5rem;
    }

    .error-file {
      font-family: monospace;
      font-size: 0.75rem;
      color: var(--text-secondary-color, #6b7280);
    }

    .empty-state {
      text-align: center;
      padding: 2rem;
      color: var(--text-secondary-color, #6b7280);
    }

    .empty-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 1rem;
      opacity: 0.5;
    }

    .start-button {
      background: var(--primary-color, #3b82f6);
      color: white;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 0.5rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin: 1rem auto 0;
    }

    .start-button:hover {
      background: var(--primary-hover-color, #2563eb);
      transform: translateY(-1px);
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .progress-container {
        padding: 1rem;
      }

      .details-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
      }

      .progress-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
      }
    }
  `;

  private setupStoreSubscriptions(): void {
    this.subscriptions = [
      this.store.onAnalysisProgress((progress: ProgressData) => {
        this.currentProgress = progress;
      }),
      
      this.store.onAnalysisStarted((session: AnalysisSession) => {
        this.session = session;
      }),
      
      this.store.onStateChanged((state) => {
        this.loadingState = state.loadingStates['analysis-start'] || { isLoading: false };
      })
    ];
  }

  private cleanupSubscriptions(): void {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.subscriptions = [];
  }

  private async handleStartAnalysis(): Promise<void> {
    if (this.projectId) {
      await this.store.startAnalysis(this.projectId);
    }
  }

  private async handleCancelAnalysis(): Promise<void> {
    if (this.session?.id) {
      await this.store.cancelAnalysis(this.session.id);
    }
  }

  private formatPhase(phase: AnalysisPhase): string {
    return phase.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  private formatDuration(seconds: number): string {
    if (seconds < 60) {
      return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.round(seconds % 60);
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  }

  private formatTime(timestamp: string): string {
    return new Date(timestamp).toLocaleTimeString();
  }

  private renderEmptyState(): any {
    return html`
      <div class="empty-state">
        <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" 
                d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
        <h3>No Analysis Running</h3>
        <p>Start analyzing your project to see progress here.</p>
        <button class="start-button" @click=${this.handleStartAnalysis}>
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z"/>
          </svg>
          Start Analysis
        </button>
      </div>
    `;
  }

  private renderProgress(): any {
    const progress = this.currentProgress;
    if (!progress) return this.renderEmptyState();

    const percentage = Math.round(progress.percentage);
    const isRunning = this.session?.status === 'running';
    const isCompleted = this.session?.status === 'completed';
    const isFailed = this.session?.status === 'failed';

    return html`
      <div class="progress-header">
        <h3 class="progress-title">
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
          </svg>
          Analysis Progress
        </h3>
        
        <div class="progress-controls">
          ${!this.compact ? html`
            <button class="control-button" @click=${() => this.showDetails = !this.showDetails}>
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
            </button>
          ` : ''}
          
          ${isRunning ? html`
            <button class="control-button danger" @click=${this.handleCancelAnalysis}>
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 6h12v12H6z"/>
              </svg>
            </button>
          ` : ''}
        </div>
      </div>

      <div class="progress-status">
        <div class="status-indicator ${isRunning ? 'running' : isCompleted ? 'completed' : isFailed ? 'failed' : ''}"></div>
        <div>
          <div class="status-text">
            ${isRunning ? 'Analyzing...' : isCompleted ? 'Analysis Complete' : isFailed ? 'Analysis Failed' : 'Analysis Queued'}
          </div>
          <div class="status-phase">${this.formatPhase(progress.current_phase)}</div>
        </div>
      </div>

      <div class="progress-bar-container">
        <div class="progress-bar ${this.compact ? 'compact' : ''}">
          <div class="progress-fill ${isCompleted ? 'completed' : isFailed ? 'failed' : ''}"
               style="width: ${percentage}%"></div>
        </div>
        <div class="progress-text">
          <span>${progress.files_processed} / ${progress.total_files} files</span>
          <span>${percentage}%</span>
        </div>
      </div>

      ${progress.current_file ? html`
        <div class="current-file">
          <div class="current-file-label">Currently Processing</div>
          <div class="current-file-path">${progress.current_file}</div>
        </div>
      ` : ''}

      ${this.showDetails ? this.renderDetails() : ''}
      ${progress.errors.length > 0 ? this.renderErrors() : ''}
    `;
  }

  private renderDetails(): any {
    const progress = this.currentProgress;
    if (!progress) return '';

    return html`
      <div class="progress-details">
        <div class="details-grid">
          <div class="detail-item">
            <div class="detail-value">${progress.speed.files_per_second.toFixed(1)}</div>
            <div class="detail-label">Files/sec</div>
          </div>
          <div class="detail-item">
            <div class="detail-value">${this.formatDuration(progress.estimated_remaining)}</div>
            <div class="detail-label">Remaining</div>
          </div>
          <div class="detail-item">
            <div class="detail-value">${progress.current_phase}</div>
            <div class="detail-label">Phase</div>
          </div>
          <div class="detail-item">
            <div class="detail-value">${progress.total_phases}</div>
            <div class="detail-label">Total Phases</div>
          </div>
        </div>
      </div>
    `;
  }

  private renderErrors(): any {
    const progress = this.currentProgress;
    if (!progress || progress.errors.length === 0) return '';

    return html`
      <div class="errors-section">
        <div class="errors-header">
          <div class="errors-title">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            Errors (${progress.errors.length})
          </div>
        </div>
        
        ${progress.errors.map(error => html`
          <div class="error-item">
            <div class="error-header">
              <span class="error-type">${error.type.toUpperCase()}</span>
              <span class="error-time">${this.formatTime(error.timestamp)}</span>
            </div>
            <div class="error-message">${error.message}</div>
            ${error.file_path ? html`
              <div class="error-file">${error.file_path}${error.line_number ? `:${error.line_number}` : ''}</div>
            ` : ''}
          </div>
        `)}
      </div>
    `;
  }

  render() {
    const containerClasses = `progress-container ${this.compact ? 'compact' : ''}`;

    return html`
      <div class=${containerClasses}>
        ${this.currentProgress || this.session ? this.renderProgress() : this.renderEmptyState()}
      </div>
    `;
  }
}
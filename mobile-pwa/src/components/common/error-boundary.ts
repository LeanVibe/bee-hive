import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'

@customElement('error-boundary')
export class ErrorBoundary extends LitElement {
  @state() private hasError: boolean = false
  @state() private errorMessage: string = ''
  @state() private errorStack: string = ''
  
  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
    }
    
    .error-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 300px;
      padding: 2rem;
      text-align: center;
    }
    
    .error-icon {
      width: 64px;
      height: 64px;
      color: #ef4444;
      margin-bottom: 1rem;
    }
    
    .error-title {
      font-size: 1.5rem;
      font-weight: 600;
      color: #1f2937;
      margin-bottom: 0.5rem;
    }
    
    .error-message {
      font-size: 1rem;
      color: #6b7280;
      margin-bottom: 1.5rem;
      max-width: 500px;
    }
    
    .error-actions {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
      justify-content: center;
    }
    
    .btn {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      min-height: 44px;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .btn-primary {
      background: #3b82f6;
      color: white;
    }
    
    .btn-primary:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }
    
    .btn-secondary {
      background: white;
      color: #374151;
      border: 1px solid #d1d5db;
    }
    
    .btn-secondary:hover {
      background: #f9fafb;
      transform: translateY(-1px);
    }
    
    .error-details {
      margin-top: 2rem;
      padding: 1rem;
      background: #f3f4f6;
      border-radius: 0.5rem;
      text-align: left;
      max-width: 600px;
      width: 100%;
    }
    
    .error-details summary {
      cursor: pointer;
      font-weight: 500;
      color: #374151;
      padding: 0.5rem;
    }
    
    .error-details pre {
      margin: 1rem 0 0 0;
      padding: 1rem;
      background: #1f2937;
      color: #f9fafb;
      border-radius: 0.25rem;
      overflow-x: auto;
      font-size: 0.75rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    
    @media (max-width: 640px) {
      .error-container {
        padding: 1rem;
      }
      
      .error-actions {
        flex-direction: column;
        width: 100%;
      }
      
      .btn {
        width: 100%;
        justify-content: center;
      }
    }
  `
  
  connectedCallback() {
    super.connectedCallback()
    
    // Listen for unhandled errors
    window.addEventListener('error', this.handleError.bind(this))
    window.addEventListener('unhandledrejection', this.handleRejection.bind(this))
  }
  
  disconnectedCallback() {
    super.disconnectedCallback()
    
    window.removeEventListener('error', this.handleError.bind(this))
    window.removeEventListener('unhandledrejection', this.handleRejection.bind(this))
  }
  
  private handleError(event: ErrorEvent) {
    this.showError(event.error || new Error(event.message))
  }
  
  private handleRejection(event: PromiseRejectionEvent) {
    this.showError(event.reason)
  }
  
  private showError(error: Error) {
    this.hasError = true
    this.errorMessage = error.message || 'An unexpected error occurred'
    this.errorStack = error.stack || ''
    
    console.error('Error boundary caught error:', error)
    
    // Report error to monitoring service
    this.reportError(error)
  }
  
  private reportError(error: Error) {
    // In a real app, this would send to error tracking service
    console.error('Reporting error:', {
      message: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent
    })
  }
  
  private handleReload() {
    window.location.reload()
  }
  
  private handleGoHome() {
    this.hasError = false
    window.location.href = '/'
  }
  
  private handleReset() {
    this.hasError = false
    this.errorMessage = ''
    this.errorStack = ''
    
    // Try to recover by re-rendering children
    this.requestUpdate()
  }
  
  render() {
    if (this.hasError) {
      return html`
        <div class="error-container">
          <svg class="error-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          
          <h2 class="error-title">Something went wrong</h2>
          <p class="error-message">
            ${this.errorMessage || 'An unexpected error occurred. Please try refreshing the page.'}
          </p>
          
          <div class="error-actions">
            <button class="btn btn-primary" @click=${this.handleReload}>
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Reload Page
            </button>
            
            <button class="btn btn-secondary" @click=${this.handleGoHome}>
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              Go Home
            </button>
            
            <button class="btn btn-secondary" @click=${this.handleReset}>
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Try Again
            </button>
          </div>
          
          ${this.errorStack ? html`
            <details class="error-details">
              <summary>Error Details (for developers)</summary>
              <pre>${this.errorStack}</pre>
            </details>
          ` : ''}
        </div>
      `
    }
    
    return html`<slot></slot>`
  }
}
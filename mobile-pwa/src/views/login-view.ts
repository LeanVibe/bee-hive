import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { AuthService } from '../services/auth'

@customElement('login-view')
export class LoginView extends LitElement {
  @state() private email: string = ''
  @state() private password: string = ''
  @state() private isLoading: boolean = false
  @state() private error: string = ''
  @state() private success: string = ''
  @state() private showWebAuthn: boolean = false
  @state() private showAuth0: boolean = false
  @state() private loginMethod: 'password' | 'auth0' | 'webauthn' = 'password'
  @state() private rememberMe: boolean = false
  
  private authService: AuthService
  
  static styles = css`
    :host {
      display: block;
      min-height: 100vh;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .login-container {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }
    
    .login-card {
      width: 100%;
      max-width: 400px;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      border-radius: 1rem;
      padding: 2rem;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .logo {
      text-align: center;
      margin-bottom: 2rem;
    }
    
    .logo-icon {
      width: 64px;
      height: 64px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 1rem;
      margin: 0 auto 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 2rem;
    }
    
    .logo-text {
      font-size: 1.5rem;
      font-weight: 700;
      color: #1f2937;
      margin: 0;
    }
    
    .logo-subtitle {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0.25rem 0 0 0;
    }
    
    .form-group {
      margin-bottom: 1.5rem;
    }
    
    .form-label {
      display: block;
      font-size: 0.875rem;
      font-weight: 500;
      color: #374151;
      margin-bottom: 0.5rem;
    }
    
    .form-input {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid #d1d5db;
      border-radius: 0.5rem;
      font-size: 1rem;
      transition: all 0.2s;
      box-sizing: border-box;
    }
    
    .form-input:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .btn {
      width: 100%;
      padding: 0.75rem;
      border: none;
      border-radius: 0.5rem;
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      min-height: 48px;
    }
    
    .btn-primary {
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      color: white;
    }
    
    .btn-primary:hover:not(:disabled) {
      background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
      transform: translateY(-1px);
    }
    
    .btn-secondary {
      background: white;
      color: #374151;
      border: 1px solid #d1d5db;
      margin-top: 0.75rem;
    }
    
    .btn-secondary:hover:not(:disabled) {
      background: #f9fafb;
      transform: translateY(-1px);
    }
    
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    
    .spinner {
      width: 20px;
      height: 20px;
      border: 2px solid transparent;
      border-top: 2px solid currentColor;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    .error-message {
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #dc2626;
      padding: 0.75rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      margin-bottom: 1rem;
    }
    
    .success-message {
      background: #f0f9f9;
      border: 1px solid #a7f3d0;
      color: #047857;
      padding: 0.75rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      margin-bottom: 1rem;
    }
    
    .login-methods {
      margin-bottom: 1.5rem;
    }
    
    .method-tabs {
      display: flex;
      border-radius: 0.5rem;
      background: #f3f4f6;
      padding: 0.25rem;
      margin-bottom: 1.5rem;
    }
    
    .method-tab {
      flex: 1;
      padding: 0.5rem 0.75rem;
      text-align: center;
      border: none;
      background: transparent;
      color: #6b7280;
      font-size: 0.875rem;
      font-weight: 500;
      border-radius: 0.375rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .method-tab.active {
      background: white;
      color: #374151;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .method-tab:hover:not(.active) {
      color: #374151;
    }
    
    .checkbox-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
    }
    
    .checkbox {
      width: 16px;
      height: 16px;
      accent-color: #3b82f6;
    }
    
    .checkbox-label {
      font-size: 0.875rem;
      color: #6b7280;
      cursor: pointer;
    }
    
    .btn-auth0 {
      background: #eb5424;
      color: white;
    }
    
    .btn-auth0:hover:not(:disabled) {
      background: #dc4914;
      transform: translateY(-1px);
    }
    
    .auth0-icon {
      width: 20px;
      height: 20px;
    }
    
    .security-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      background: #ecfdf5;
      color: #065f46;
      padding: 0.25rem 0.5rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 500;
      margin-bottom: 1rem;
    }
    
    .enterprise-features {
      margin-top: 1.5rem;
      padding-top: 1.5rem;
      border-top: 1px solid #e5e7eb;
    }
    
    .enterprise-title {
      font-size: 1rem;
      font-weight: 600;
      color: #1f2937;
      margin-bottom: 1rem;
      text-align: center;
    }
    
    .divider {
      margin: 1.5rem 0;
      text-align: center;
      position: relative;
      color: #6b7280;
      font-size: 0.875rem;
    }
    
    .divider::before {
      content: '';
      position: absolute;
      top: 50%;
      left: 0;
      right: 0;
      height: 1px;
      background: #e5e7eb;
    }
    
    .divider span {
      background: rgba(255, 255, 255, 0.95);
      padding: 0 1rem;
    }
    
    .webauthn-icon {
      width: 20px;
      height: 20px;
    }
    
    .feature-list {
      margin-top: 2rem;
      padding-top: 1.5rem;
      border-top: 1px solid #e5e7eb;
    }
    
    .feature-item {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
      font-size: 0.875rem;
      color: #6b7280;
    }
    
    .feature-icon {
      width: 16px;
      height: 16px;
      color: #10b981;
    }
    
    @media (max-width: 640px) {
      .login-container {
        padding: 0.5rem;
      }
      
      .login-card {
        padding: 1.5rem;
      }
    }
  `
  
  constructor() {
    super()
    this.authService = AuthService.getInstance()
    this.checkAuthenticationCapabilities()
  }
  
  private async checkAuthenticationCapabilities() {
    // Check WebAuthn support
    try {
      if (window.PublicKeyCredential && 
          await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable()) {
        this.showWebAuthn = true
      }
    } catch (error) {
      console.log('WebAuthn not available:', error)
    }
    
    // Check Auth0 availability
    const auth0Config = this.authService.getAuth0Config()
    this.showAuth0 = !!(auth0Config.domain && auth0Config.clientId && 
                       auth0Config.domain !== 'your-domain.auth0.com')
  }
  
  private handleEmailChange(e: Event) {
    this.email = (e.target as HTMLInputElement).value
  }
  
  private handlePasswordChange(e: Event) {
    this.password = (e.target as HTMLInputElement).value
  }
  
  private async handleSubmit(e: Event) {
    e.preventDefault()
    
    if (!this.email || !this.password) {
      this.error = 'Please enter both email and password'
      return
    }
    
    this.isLoading = true
    this.error = ''
    
    try {
      await this.authService.login({
        email: this.email,
        password: this.password
      })
      
      // Navigation will be handled by the auth service event
      
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Login failed'
    } finally {
      this.isLoading = false
    }
  }
  
  private async handleWebAuthnLogin() {
    this.isLoading = true
    this.error = ''
    
    try {
      await this.authService.loginWithWebAuthn()
      
      // Navigation will be handled by the auth service event
      
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'WebAuthn login failed'
    } finally {
      this.isLoading = false
    }
  }
  
  private async handleAuth0Login() {
    this.isLoading = true
    this.error = ''
    
    try {
      await this.authService.loginWithAuth0()
      // This will redirect to Auth0, so we won't reach here normally
      
    } catch (error) {
      if ((error as Error).message === 'Redirect in progress') {
        // This is expected - user is being redirected to Auth0
        return
      }
      this.error = (error as Error).message || 'SSO login failed'
      this.isLoading = false
    }
  }
  
  private handleRememberMeChange(e: Event) {
    this.rememberMe = (e.target as HTMLInputElement).checked
  }
  
  private setLoginMethod(method: 'password' | 'auth0' | 'webauthn') {
    this.loginMethod = method
    this.error = ''
    this.success = ''
  }
  
  render() {
    const hasMultipleMethods = this.showAuth0 || this.showWebAuthn
    
    return html`
      <div class="login-container">
        <div class="login-card">
          <div class="logo">
            <div class="logo-icon">🤖</div>
            <h1 class="logo-text">Agent Hive</h1>
            <p class="logo-subtitle">Enterprise Autonomous Development Platform</p>
          </div>
          
          <div class="security-badge">
            <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 1l3.09 6.26L22 9l-5 4.87L18.18 21 12 17.77 5.82 21 7 13.87 2 9l6.91-1.74L12 1z"/>
            </svg>
            Enterprise Security Enabled
          </div>
          
          ${this.error ? html`
            <div class="error-message">
              ${this.error}
            </div>
          ` : ''}
          
          ${this.success ? html`
            <div class="success-message">
              ${this.success}
            </div>
          ` : ''}
          
          ${hasMultipleMethods ? html`
            <div class="login-methods">
              <div class="method-tabs">
                <button
                  type="button"
                  class="method-tab ${this.loginMethod === 'password' ? 'active' : ''}"
                  @click=${() => this.setLoginMethod('password')}
                  ?disabled=${this.isLoading}
                >
                  Password
                </button>
                
                ${this.showAuth0 ? html`
                  <button
                    type="button"
                    class="method-tab ${this.loginMethod === 'auth0' ? 'active' : ''}"
                    @click=${() => this.setLoginMethod('auth0')}
                    ?disabled=${this.isLoading}
                  >
                    Enterprise SSO
                  </button>
                ` : ''}
                
                ${this.showWebAuthn ? html`
                  <button
                    type="button"
                    class="method-tab ${this.loginMethod === 'webauthn' ? 'active' : ''}"
                    @click=${() => this.setLoginMethod('webauthn')}
                    ?disabled=${this.isLoading}
                  >
                    Biometric
                  </button>
                ` : ''}
              </div>
            </div>
          ` : ''}
          
          ${this.loginMethod === 'password' ? html`
            <form @submit=${this.handleSubmit}>
              <div class="form-group">
                <label class="form-label" for="email">Email</label>
                <input
                  id="email"
                  type="email"
                  class="form-input"
                  .value=${this.email}
                  @input=${this.handleEmailChange}
                  required
                  autocomplete="email"
                  ?disabled=${this.isLoading}
                />
              </div>
              
              <div class="form-group">
                <label class="form-label" for="password">Password</label>
                <input
                  id="password"
                  type="password"
                  class="form-input"
                  .value=${this.password}
                  @input=${this.handlePasswordChange}
                  required
                  autocomplete="current-password"
                  ?disabled=${this.isLoading}
                />
              </div>
              
              <div class="checkbox-group">
                <input
                  id="remember"
                  type="checkbox"
                  class="checkbox"
                  .checked=${this.rememberMe}
                  @change=${this.handleRememberMeChange}
                  ?disabled=${this.isLoading}
                />
                <label for="remember" class="checkbox-label">Remember me for 30 days</label>
              </div>
              
              <button
                type="submit"
                class="btn btn-primary"
                ?disabled=${this.isLoading}
              >
                ${this.isLoading ? html`
                  <div class="spinner"></div>
                  Signing in...
                ` : html`
                  Sign In
                `}
              </button>
            </form>
          ` : ''}
          
          ${this.loginMethod === 'auth0' ? html`
            <button
              type="button"
              class="btn btn-auth0"
              @click=${this.handleAuth0Login}
              ?disabled=${this.isLoading}
            >
              ${this.isLoading ? html`
                <div class="spinner"></div>
                Redirecting...
              ` : html`
                <svg class="auth0-icon" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M21.98 7.448L19.62 0H4.347L2.02 7.448c-1.352 4.312.03 9.206 3.815 12.015L12.007 24l6.157-4.537c3.785-2.809 5.167-7.703 3.815-12.015z"/>
                </svg>
                Continue with Enterprise SSO
              `}
            </button>
            <p style="font-size: 0.875rem; color: #6b7280; text-align: center; margin-top: 0.75rem;">
              Redirecting to your organization's single sign-on portal
            </p>
          ` : ''}
          
          ${this.loginMethod === 'webauthn' ? html`
            <button
              type="button"
              class="btn btn-secondary"
              @click=${this.handleWebAuthnLogin}
              ?disabled=${this.isLoading}
            >
              ${this.isLoading ? html`
                <div class="spinner"></div>
                Authenticating...
              ` : html`
                <svg class="webauthn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                Use Biometric Authentication
              `}
            </button>
            <p style="font-size: 0.875rem; color: #6b7280; text-align: center; margin-top: 0.75rem;">
              Use your fingerprint, face, or security key to sign in
            </p>
          ` : ''}
          
          <div class="enterprise-features">
            <h3 class="enterprise-title">Enterprise Platform Features</h3>
            <div class="feature-item">
              <svg class="feature-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Multi-agent orchestration & monitoring
            </div>
            <div class="feature-item">
              <svg class="feature-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Real-time performance analytics
            </div>
            <div class="feature-item">
              <svg class="feature-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
              Enterprise security & compliance
            </div>
            <div class="feature-item">
              <svg class="feature-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
              Cross-platform PWA dashboard
            </div>
          </div>
        </div>
      </div>
    `
  }
}
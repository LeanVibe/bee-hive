import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'

@customElement('install-prompt')
export class InstallPrompt extends LitElement {
  @state() declare private showPrompt: boolean
  @state() declare private isInstalled: boolean
  private deferredPrompt: any = null
  
  constructor() {
    super()
    
    // Initialize state properties
    this.showPrompt = false
    this.isInstalled = false
  }
  
  static styles = css`
    :host {
      display: block;
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      z-index: 50;
      transform: translateY(100%);
      transition: transform 0.3s ease-out;
    }
    
    :host(.show) {
      transform: translateY(0);
    }
    
    .prompt {
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      color: white;
      padding: 1rem;
      margin: 1rem;
      border-radius: 1rem;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    
    .prompt-content {
      flex: 1;
    }
    
    .prompt-title {
      font-size: 1rem;
      font-weight: 600;
      margin: 0 0 0.25rem 0;
    }
    
    .prompt-text {
      font-size: 0.875rem;
      opacity: 0.9;
      margin: 0;
    }
    
    .prompt-actions {
      display: flex;
      gap: 0.75rem;
      align-items: center;
    }
    
    .btn {
      padding: 0.5rem 1rem;
      border: none;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      min-height: 36px;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .btn-primary {
      background: rgba(255, 255, 255, 0.2);
      color: white;
      backdrop-filter: blur(10px);
    }
    
    .btn-primary:hover {
      background: rgba(255, 255, 255, 0.3);
      transform: translateY(-1px);
    }
    
    .btn-ghost {
      background: transparent;
      color: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .btn-ghost:hover {
      color: white;
      border-color: rgba(255, 255, 255, 0.5);
    }
    
    .close-btn {
      background: transparent;
      border: none;
      color: rgba(255, 255, 255, 0.8);
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 0.25rem;
      transition: color 0.2s;
    }
    
    .close-btn:hover {
      color: white;
    }
    
    .install-icon {
      width: 40px;
      height: 40px;
      background: rgba(255, 255, 255, 0.2);
      border-radius: 0.75rem;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
    }
    
    @supports (padding: max(0px)) {
      .prompt {
        margin-bottom: max(1rem, env(safe-area-inset-bottom));
      }
    }
    
    @media (max-width: 640px) {
      .prompt {
        margin: 0.5rem;
        flex-direction: column;
        text-align: center;
        gap: 1rem;
      }
      
      .prompt-actions {
        width: 100%;
        justify-content: center;
      }
      
      .btn {
        flex: 1;
        justify-content: center;
      }
    }
  `
  
  connectedCallback() {
    super.connectedCallback()
    this.setupInstallPrompt()
    this.checkInstallStatus()
  }
  
  private setupInstallPrompt() {
    // Listen for the beforeinstallprompt event
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault()
      this.deferredPrompt = e
      this.showInstallPrompt()
    })
    
    // Listen for successful installation
    window.addEventListener('appinstalled', () => {
      this.isInstalled = true
      this.hideInstallPrompt()
      this.showInstalledMessage()
    })
  }
  
  private checkInstallStatus() {
    // Check if app is already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      this.isInstalled = true
      return
    }
    
    // Check if running in installed PWA
    if ((navigator as any).standalone) {
      this.isInstalled = true
      return
    }
    
    // Check if user has previously dismissed the prompt
    const dismissed = localStorage.getItem('install-prompt-dismissed')
    if (dismissed) {
      const dismissedTime = parseInt(dismissed, 10)
      const daysSinceDismissal = (Date.now() - dismissedTime) / (1000 * 60 * 60 * 24)
      
      // Show again after 7 days
      if (daysSinceDismissal < 7) {
        return
      }
    }
    
    // Show prompt after a delay if not installed and criteria met
    setTimeout(() => {
      if (!this.isInstalled && this.shouldShowPrompt()) {
        this.showInstallPrompt()
      }
    }, 3000) // Show after 3 seconds
  }
  
  private shouldShowPrompt(): boolean {
    // Only show on mobile devices
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    
    // Only show if PWA is installable
    const isInstallable = this.deferredPrompt !== null
    
    // Don't show if already installed
    if (this.isInstalled) return false
    
    return isMobile || isInstallable
  }
  
  private showInstallPrompt() {
    this.showPrompt = true
    this.classList.add('show')
  }
  
  private hideInstallPrompt() {
    this.showPrompt = false
    this.classList.remove('show')
  }
  
  private async handleInstall() {
    if (!this.deferredPrompt) {
      // Fallback for browsers that don't support the API
      this.showManualInstallInstructions()
      return
    }
    
    try {
      // Show the install prompt
      this.deferredPrompt.prompt()
      
      // Wait for user response
      const { outcome } = await this.deferredPrompt.userChoice
      
      if (outcome === 'accepted') {
        console.log('User accepted the install prompt')
      } else {
        console.log('User dismissed the install prompt')
        this.handleDismiss()
      }
      
      // Clear the deferred prompt
      this.deferredPrompt = null
      this.hideInstallPrompt()
      
    } catch (error) {
      console.error('Install prompt failed:', error)
      this.showManualInstallInstructions()
    }
  }
  
  private handleDismiss() {
    // Remember that user dismissed the prompt
    localStorage.setItem('install-prompt-dismissed', Date.now().toString())
    this.hideInstallPrompt()
  }
  
  private showManualInstallInstructions() {
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
    const isAndroid = /Android/.test(navigator.userAgent)
    
    let instructions = ''
    
    if (isIOS) {
      instructions = 'Tap the Share button in Safari and select "Add to Home Screen"'
    } else if (isAndroid) {
      instructions = 'Tap the menu button in your browser and select "Add to Home Screen" or "Install App"'
    } else {
      instructions = 'Look for the install button in your browser\'s address bar or menu'
    }
    
    // Show a toast or modal with instructions
    this.showToast(instructions)
    this.hideInstallPrompt()
  }
  
  private showToast(message: string) {
    // Simple toast implementation
    const toast = document.createElement('div')
    toast.style.cssText = `
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: #1f2937;
      color: white;
      padding: 1rem 1.5rem;
      border-radius: 0.5rem;
      z-index: 9999;
      font-size: 0.875rem;
      max-width: 90%;
      text-align: center;
    `
    toast.textContent = message
    
    document.body.appendChild(toast)
    
    setTimeout(() => {
      document.body.removeChild(toast)
    }, 5000)
  }
  
  private showInstalledMessage() {
    this.showToast('App installed successfully! 🎉')
  }
  
  render() {
    if (!this.showPrompt || this.isInstalled) {
      return html``
    }
    
    return html`
      <div class="prompt">
        <div class="install-icon">📱</div>
        
        <div class="prompt-content">
          <h3 class="prompt-title">Install Agent Hive</h3>
          <p class="prompt-text">
            Install the app for the best mobile experience with offline support and push notifications.
          </p>
        </div>
        
        <div class="prompt-actions">
          <button class="btn btn-primary" @click=${this.handleInstall}>
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Install
          </button>
          
          <button class="btn btn-ghost" @click=${this.handleDismiss}>
            Not now
          </button>
        </div>
        
        <button class="close-btn" @click=${this.handleDismiss} aria-label="Close">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    `
  }
}
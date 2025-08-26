/**
 * Enhanced Mobile Experience Component
 * 
 * Provides advanced mobile-specific features for the Phase 4 PWA:
 * - Gesture navigation
 * - Pull-to-refresh functionality
 * - Haptic feedback
 * - Voice commands
 * - Offline sync indicators
 * - Touch-optimized interactions
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface MobileGesture {
  type: 'swipe' | 'pinch' | 'tap' | 'long-press';
  direction?: 'left' | 'right' | 'up' | 'down';
  velocity?: number;
  distance?: number;
  duration?: number;
}

interface VoiceCommand {
  phrase: string;
  action: string;
  confidence: number;
}

@customElement('enhanced-mobile-experience')
export class EnhancedMobileExperience extends LitElement {
  @property({ type: Boolean }) enabled = true;
  @property({ type: Boolean }) voiceEnabled = true;
  @property({ type: Boolean }) hapticsEnabled = true;
  @property({ type: Boolean }) gesturesEnabled = true;
  @property({ type: Boolean }) pullToRefreshEnabled = true;

  @state() private isListening = false;
  @state() private lastGesture: MobileGesture | null = null;
  @state() private refreshing = false;
  @state() private offlineQueueSize = 0;

  private recognition: any;
  private touchStartY = 0;
  private touchStartTime = 0;
  private gestureThreshold = 50;
  private velocityThreshold = 0.5;

  static styles = css`
    :host {
      display: contents;
      pointer-events: none;
    }

    .mobile-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      pointer-events: none;
      z-index: 1000;
    }

    .gesture-indicator {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: rgba(59, 130, 246, 0.9);
      color: white;
      padding: 0.5rem 1rem;
      border-radius: 2rem;
      font-size: 0.875rem;
      font-weight: 500;
      transform: translateY(100px);
      opacity: 0;
      transition: all 0.3s ease;
      pointer-events: auto;
      backdrop-filter: blur(10px);
    }

    .gesture-indicator.visible {
      transform: translateY(0);
      opacity: 1;
    }

    .voice-button {
      position: fixed;
      bottom: 80px;
      right: 20px;
      width: 60px;
      height: 60px;
      background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
      border: none;
      border-radius: 50%;
      color: white;
      font-size: 1.5rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
      transition: all 0.3s ease;
      pointer-events: auto;
      z-index: 1001;
    }

    .voice-button:hover {
      transform: scale(1.1);
      box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6);
    }

    .voice-button.listening {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      animation: pulse 1.5s ease-in-out infinite;
    }

    .voice-button.disabled {
      background: #6b7280;
      cursor: not-allowed;
    }

    .pull-refresh-indicator {
      position: fixed;
      top: -60px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(255, 255, 255, 0.95);
      border-radius: 2rem;
      padding: 1rem 2rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-weight: 500;
      color: #374151;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      transition: transform 0.3s ease;
      backdrop-filter: blur(10px);
      z-index: 1002;
    }

    .pull-refresh-indicator.visible {
      transform: translateX(-50%) translateY(80px);
    }

    .pull-refresh-indicator.refreshing {
      transform: translateX(-50%) translateY(80px);
    }

    .refresh-spinner {
      width: 20px;
      height: 20px;
      border: 2px solid #e5e7eb;
      border-top: 2px solid #3b82f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .offline-indicator {
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(245, 158, 11, 0.95);
      color: white;
      padding: 0.75rem 1.5rem;
      border-radius: 2rem;
      font-size: 0.875rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
      z-index: 1003;
      pointer-events: auto;
    }

    .offline-indicator.connected {
      background: rgba(16, 185, 129, 0.95);
      box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }

    .haptic-feedback {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 100px;
      height: 100px;
      background: rgba(59, 130, 246, 0.1);
      border: 3px solid rgba(59, 130, 246, 0.3);
      border-radius: 50%;
      opacity: 0;
      pointer-events: none;
      z-index: 999;
    }

    .haptic-feedback.active {
      animation: hapticPulse 0.6s ease-out;
    }

    .voice-commands-help {
      position: fixed;
      bottom: 150px;
      right: 20px;
      background: rgba(255, 255, 255, 0.95);
      border-radius: 1rem;
      padding: 1rem;
      max-width: 300px;
      font-size: 0.875rem;
      color: #374151;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      transform: translateX(100%);
      opacity: 0;
      transition: all 0.3s ease;
      backdrop-filter: blur(10px);
      z-index: 1000;
      pointer-events: auto;
    }

    .voice-commands-help.visible {
      transform: translateX(0);
      opacity: 1;
    }

    .voice-commands-help h4 {
      margin: 0 0 0.5rem 0;
      font-weight: 600;
      color: #111827;
    }

    .voice-commands-help ul {
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .voice-commands-help li {
      padding: 0.25rem 0;
      display: flex;
      justify-content: space-between;
    }

    .command-phrase {
      font-weight: 500;
      color: #3b82f6;
    }

    @keyframes pulse {
      0%, 100% {
        transform: scale(1);
      }
      50% {
        transform: scale(1.1);
      }
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    @keyframes hapticPulse {
      0% {
        opacity: 0;
        transform: translate(-50%, -50%) scale(0.5);
      }
      50% {
        opacity: 1;
        transform: translate(-50%, -50%) scale(1);
      }
      100% {
        opacity: 0;
        transform: translate(-50%, -50%) scale(1.5);
      }
    }

    /* Hide on desktop */
    @media (min-width: 769px) {
      .voice-button,
      .gesture-indicator,
      .pull-refresh-indicator,
      .voice-commands-help {
        display: none;
      }
    }

    /* Enhanced touch targets for mobile */
    @media (max-width: 768px) {
      .voice-button {
        width: 56px;
        height: 56px;
        bottom: 70px;
        right: 16px;
      }

      .gesture-indicator {
        bottom: 16px;
        right: 16px;
        font-size: 0.8125rem;
        padding: 0.375rem 0.75rem;
      }
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    if (this.enabled) {
      this.setupMobileFeatures();
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.cleanupMobileFeatures();
  }

  private setupMobileFeatures() {
    // Set up gesture handling
    if (this.gesturesEnabled) {
      this.setupGestureHandling();
    }

    // Set up voice recognition
    if (this.voiceEnabled && 'webkitSpeechRecognition' in window) {
      this.setupVoiceRecognition();
    }

    // Set up pull-to-refresh
    if (this.pullToRefreshEnabled) {
      this.setupPullToRefresh();
    }

    // Listen for network changes
    window.addEventListener('online', this.handleOnline.bind(this));
    window.addEventListener('offline', this.handleOffline.bind(this));
  }

  private cleanupMobileFeatures() {
    if (this.recognition) {
      this.recognition.stop();
    }

    window.removeEventListener('online', this.handleOnline.bind(this));
    window.removeEventListener('offline', this.handleOffline.bind(this));
  }

  private setupGestureHandling() {
    document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
    document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
    document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
  }

  private handleTouchStart(event: TouchEvent) {
    if (event.touches.length === 1) {
      this.touchStartY = event.touches[0].clientY;
      this.touchStartTime = Date.now();
    }
  }

  private handleTouchMove(event: TouchEvent) {
    if (event.touches.length === 1 && this.pullToRefreshEnabled) {
      const touch = event.touches[0];
      const deltaY = touch.clientY - this.touchStartY;
      
      // Show pull-to-refresh indicator when pulling down from top
      if (deltaY > this.gestureThreshold && window.scrollY === 0) {
        event.preventDefault();
        this.showPullToRefreshIndicator(true);
      }
    }
  }

  private handleTouchEnd(event: TouchEvent) {
    if (event.changedTouches.length === 1) {
      const touch = event.changedTouches[0];
      const deltaY = touch.clientY - this.touchStartY;
      const deltaTime = Date.now() - this.touchStartTime;
      const velocity = Math.abs(deltaY) / deltaTime;

      // Detect gestures
      if (Math.abs(deltaY) > this.gestureThreshold && velocity > this.velocityThreshold) {
        const gesture: MobileGesture = {
          type: 'swipe',
          direction: deltaY > 0 ? 'down' : 'up',
          velocity,
          distance: Math.abs(deltaY),
          duration: deltaTime
        };

        this.handleGesture(gesture);
      }

      // Handle pull-to-refresh
      if (deltaY > this.gestureThreshold * 2 && window.scrollY === 0) {
        this.triggerRefresh();
      } else {
        this.showPullToRefreshIndicator(false);
      }
    }
  }

  private handleGesture(gesture: MobileGesture) {
    this.lastGesture = gesture;
    this.showGestureIndicator(gesture);
    this.triggerHapticFeedback();

    // Emit gesture event
    this.dispatchEvent(new CustomEvent('mobile-gesture', {
      detail: gesture,
      bubbles: true,
      composed: true
    }));

    // Handle specific gestures
    switch (gesture.type) {
      case 'swipe':
        if (gesture.direction === 'down' && gesture.velocity > 1.0) {
          this.triggerRefresh();
        }
        break;
    }
  }

  private setupVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window)) {
      console.warn('Speech recognition not supported');
      return;
    }

    this.recognition = new (window as any).webkitSpeechRecognition();
    this.recognition.continuous = false;
    this.recognition.interimResults = false;
    this.recognition.lang = 'en-US';

    this.recognition.onstart = () => {
      this.isListening = true;
    };

    this.recognition.onend = () => {
      this.isListening = false;
    };

    this.recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript.toLowerCase();
      const confidence = event.results[0][0].confidence;

      const command: VoiceCommand = {
        phrase: transcript,
        action: this.parseVoiceCommand(transcript),
        confidence
      };

      this.handleVoiceCommand(command);
    };

    this.recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      this.isListening = false;
    };
  }

  private parseVoiceCommand(transcript: string): string {
    const commands: Record<string, string> = {
      'refresh': 'refresh',
      'reload': 'refresh',
      'update': 'refresh',
      'go to dashboard': 'navigate-dashboard',
      'show agents': 'navigate-agents',
      'show tasks': 'navigate-tasks',
      'show events': 'navigate-events',
      'show performance': 'navigate-performance',
      'phase four': 'navigate-phase4',
      'enterprise dashboard': 'navigate-phase4',
      'help': 'show-help',
      'voice commands': 'show-voice-help'
    };

    for (const [phrase, action] of Object.entries(commands)) {
      if (transcript.includes(phrase)) {
        return action;
      }
    }

    return 'unknown';
  }

  private handleVoiceCommand(command: VoiceCommand) {
    this.triggerHapticFeedback();

    // Emit voice command event
    this.dispatchEvent(new CustomEvent('voice-command', {
      detail: command,
      bubbles: true,
      composed: true
    }));

    // Handle specific commands
    switch (command.action) {
      case 'refresh':
        this.triggerRefresh();
        break;
      case 'navigate-dashboard':
        this.navigateTo('overview');
        break;
      case 'navigate-agents':
        this.navigateTo('agents');
        break;
      case 'navigate-tasks':
        this.navigateTo('kanban');
        break;
      case 'navigate-events':
        this.navigateTo('events');
        break;
      case 'navigate-performance':
        this.navigateTo('performance');
        break;
      case 'navigate-phase4':
        this.navigateTo('phase4');
        break;
      case 'show-voice-help':
        this.toggleVoiceHelp();
        break;
    }

    // Show gesture indicator with voice feedback
    this.showGestureIndicator({
      type: 'tap',
      duration: 0
    }, `Voice: ${command.phrase}`);
  }

  private navigateTo(view: string) {
    this.dispatchEvent(new CustomEvent('navigate', {
      detail: { view },
      bubbles: true,
      composed: true
    }));
  }

  private setupPullToRefresh() {
    // Pull-to-refresh is handled in touch events
  }

  private triggerRefresh() {
    if (this.refreshing) return;

    this.refreshing = true;
    this.showPullToRefreshIndicator(true);
    this.triggerHapticFeedback();

    // Emit refresh event
    this.dispatchEvent(new CustomEvent('mobile-refresh', {
      bubbles: true,
      composed: true
    }));

    // Auto-hide after 2 seconds
    setTimeout(() => {
      this.refreshing = false;
      this.showPullToRefreshIndicator(false);
    }, 2000);
  }

  private triggerHapticFeedback() {
    if (!this.hapticsEnabled) return;

    // Trigger device vibration if supported
    if ('vibrate' in navigator) {
      navigator.vibrate([50]);
    }

    // Show visual haptic feedback
    const hapticElement = this.shadowRoot?.querySelector('.haptic-feedback');
    if (hapticElement) {
      hapticElement.classList.add('active');
      setTimeout(() => {
        hapticElement.classList.remove('active');
      }, 600);
    }
  }

  private showGestureIndicator(gesture: MobileGesture, customMessage?: string) {
    const indicator = this.shadowRoot?.querySelector('.gesture-indicator') as HTMLElement;
    if (!indicator) return;

    const message = customMessage || this.getGestureMessage(gesture);
    indicator.textContent = message;
    indicator.classList.add('visible');

    setTimeout(() => {
      indicator.classList.remove('visible');
    }, 2000);
  }

  private getGestureMessage(gesture: MobileGesture): string {
    switch (gesture.type) {
      case 'swipe':
        return `Swipe ${gesture.direction}`;
      case 'tap':
        return 'Tap detected';
      case 'long-press':
        return 'Long press';
      case 'pinch':
        return 'Pinch gesture';
      default:
        return 'Gesture detected';
    }
  }

  private showPullToRefreshIndicator(visible: boolean) {
    const indicator = this.shadowRoot?.querySelector('.pull-refresh-indicator') as HTMLElement;
    if (!indicator) return;

    if (visible) {
      indicator.classList.add('visible');
      if (this.refreshing) {
        indicator.classList.add('refreshing');
      }
    } else {
      indicator.classList.remove('visible', 'refreshing');
    }
  }

  private toggleVoiceRecognition() {
    if (!this.recognition) return;

    if (this.isListening) {
      this.recognition.stop();
    } else {
      this.recognition.start();
    }
  }

  private toggleVoiceHelp() {
    const help = this.shadowRoot?.querySelector('.voice-commands-help') as HTMLElement;
    if (!help) return;

    help.classList.toggle('visible');

    // Auto-hide after 10 seconds
    setTimeout(() => {
      help.classList.remove('visible');
    }, 10000);
  }

  private handleOnline() {
    this.offlineQueueSize = 0;
    this.requestUpdate();
  }

  private handleOffline() {
    // Simulate offline queue growth
    this.offlineQueueSize = Math.floor(Math.random() * 5) + 1;
    this.requestUpdate();
  }

  render() {
    if (!this.enabled) {
      return html``;
    }

    return html`
      <div class="mobile-overlay">
        <!-- Gesture Indicator -->
        <div class="gesture-indicator">
          Gesture detected
        </div>

        <!-- Voice Recognition Button -->
        ${this.voiceEnabled && this.recognition ? html`
          <button
            class="voice-button ${this.isListening ? 'listening' : ''}"
            @click=${this.toggleVoiceRecognition}
            title="Voice commands"
            aria-label="Activate voice commands"
          >
            ${this.isListening ? '🎤' : '🎙️'}
          </button>
        ` : ''}

        <!-- Pull-to-Refresh Indicator -->
        <div class="pull-refresh-indicator">
          <div class="refresh-spinner"></div>
          ${this.refreshing ? 'Refreshing...' : 'Pull to refresh'}
        </div>

        <!-- Offline Indicator -->
        <div class="offline-indicator ${navigator.onLine ? 'connected' : ''}">
          <div style="width: 8px; height: 8px; border-radius: 50%; background: currentColor;"></div>
          ${navigator.onLine 
            ? 'Connected' 
            : this.offlineQueueSize > 0 
              ? `Offline (${this.offlineQueueSize} pending)` 
              : 'Offline'}
        </div>

        <!-- Haptic Feedback Indicator -->
        <div class="haptic-feedback"></div>

        <!-- Voice Commands Help -->
        ${this.voiceEnabled && this.recognition ? html`
          <div class="voice-commands-help">
            <h4>Voice Commands</h4>
            <ul>
              <li>
                <span class="command-phrase">"Refresh"</span>
                <span>Reload data</span>
              </li>
              <li>
                <span class="command-phrase">"Show agents"</span>
                <span>Agent view</span>
              </li>
              <li>
                <span class="command-phrase">"Show tasks"</span>
                <span>Task board</span>
              </li>
              <li>
                <span class="command-phrase">"Phase four"</span>
                <span>Enterprise dashboard</span>
              </li>
            </ul>
          </div>
        ` : ''}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'enhanced-mobile-experience': EnhancedMobileExperience;
  }
}
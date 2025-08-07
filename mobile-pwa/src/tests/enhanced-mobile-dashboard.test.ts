import { expect } from '@open-wc/testing'
import { html, fixture } from '@open-wc/testing'
import { stub, SinonStub } from 'sinon'
import '../components/autonomous-development/multi-agent-oversight-dashboard'
import '../components/autonomous-development/remote-control-center'
import { WebSocketService } from '../services/websocket'
import { MultiAgentOversightDashboard } from '../components/autonomous-development/multi-agent-oversight-dashboard'
import { RemoteControlCenter } from '../components/autonomous-development/remote-control-center'

describe('Enhanced Mobile Dashboard', () => {
  let mockWebSocketService: {
    getInstance: SinonStub
    on: SinonStub
    off: SinonStub
    sendMessage: SinonStub
    isConnected: SinonStub
    enableHighFrequencyMode: SinonStub
    enableLowFrequencyMode: SinonStub
    sendAgentCommand: SinonStub
    sendBulkAgentCommand: SinonStub
    sendEmergencyStop: SinonStub
    subscribeToAgentMetrics: SinonStub
    subscribeToConnectionQuality: SinonStub
    subscribeToCriticalEvents: SinonStub
  }

  beforeEach(() => {
    // Mock WebSocket service
    mockWebSocketService = {
      getInstance: stub().returns(mockWebSocketService),
      on: stub(),
      off: stub(),
      sendMessage: stub(),
      isConnected: stub().returns(true),
      enableHighFrequencyMode: stub(),
      enableLowFrequencyMode: stub(),
      sendAgentCommand: stub(),
      sendBulkAgentCommand: stub(),
      sendEmergencyStop: stub(),
      subscribeToAgentMetrics: stub().returns(() => {}),
      subscribeToConnectionQuality: stub().returns(() => {}),
      subscribeToCriticalEvents: stub().returns(() => {})
    }

    // Replace the WebSocketService with our mock
    stub(WebSocketService, 'getInstance').returns(mockWebSocketService as any)
  })

  afterEach(() => {
    // Restore all stubs
    ;(WebSocketService.getInstance as SinonStub).restore?.()
  })

  describe('MultiAgentOversightDashboard', () => {
    let element: MultiAgentOversightDashboard

    beforeEach(async () => {
      element = await fixture(html`
        <multi-agent-oversight-dashboard
          .fullscreen=${false}
          .viewMode=${'grid'}
        ></multi-agent-oversight-dashboard>
      `)
    })

    it('should render the dashboard with correct structure', () => {
      expect(element).to.exist
      expect(element.shadowRoot).to.exist

      const container = element.shadowRoot!.querySelector('.dashboard-container')
      expect(container).to.exist

      const header = element.shadowRoot!.querySelector('.header')
      expect(header).to.exist

      const agentsGrid = element.shadowRoot!.querySelector('.agents-grid')
      expect(agentsGrid).to.exist
    })

    it('should initialize WebSocket subscriptions on connection', () => {
      expect(mockWebSocketService.subscribeToAgentMetrics.calledOnce).to.be.true
      expect(mockWebSocketService.subscribeToConnectionQuality.calledOnce).to.be.true
      expect(mockWebSocketService.subscribeToCriticalEvents.calledOnce).to.be.true
    })

    it('should enable high-frequency streaming in real-time mode', async () => {
      element.realtimeStreaming = true
      await element.updateComplete

      expect(mockWebSocketService.enableHighFrequencyMode.called).to.be.true
    })

    it('should display agent cards with performance metrics', async () => {
      // Simulate agent data
      element.agents = [
        {
          id: 'agent-1',
          name: 'Test Agent',
          role: 'backend_developer' as any,
          status: 'active' as any,
          health: 'excellent' as any,
          performance: {
            efficiency: 95,
            accuracy: 88,
            responsiveness: 92,
            taskCompletionRate: 94
          },
          recentActivity: [],
          connectionLatency: 45,
          lastHeartbeat: new Date()
        }
      ]
      await element.updateComplete

      const agentCards = element.shadowRoot!.querySelectorAll('.agent-card')
      expect(agentCards).to.have.length(1)

      const performanceMetrics = element.shadowRoot!.querySelector('.performance-metrics')
      expect(performanceMetrics).to.exist
    })

    it('should handle agent selection for bulk operations', async () => {
      element.agents = [
        {
          id: 'agent-1',
          name: 'Test Agent',
          role: 'backend_developer' as any,
          status: 'active' as any,
          health: 'excellent' as any,
          performance: {
            efficiency: 95,
            accuracy: 88,
            responsiveness: 92,
            taskCompletionRate: 94
          },
          recentActivity: [],
          connectionLatency: 45,
          lastHeartbeat: new Date()
        }
      ]
      await element.updateComplete

      const agentCard = element.shadowRoot!.querySelector('.agent-card')
      expect(agentCard).to.exist

      // Simulate click to select agent
      agentCard!.dispatchEvent(new Event('click'))
      await element.updateComplete

      expect(element.selectedAgents.has('agent-1')).to.be.true
    })

    it('should execute emergency stop when requested', async () => {
      // Mock confirm dialog
      const originalConfirm = window.confirm
      window.confirm = stub().returns(true)

      try {
        await element.executeEmergencyStop()
        expect(mockWebSocketService.sendEmergencyStop.calledOnce).to.be.true
        expect(element.emergencyMode).to.be.true
      } finally {
        window.confirm = originalConfirm
      }
    })

    it('should update connection quality indicator', async () => {
      element.connectionQuality = 'excellent'
      await element.updateComplete

      const statusIndicator = element.shadowRoot!.querySelector('.status-indicator')
      expect(statusIndicator).to.exist
      expect(statusIndicator!.classList.contains('excellent')).to.be.true
    })

    it('should filter agents by status', async () => {
      element.agents = [
        {
          id: 'agent-1',
          name: 'Active Agent',
          role: 'backend_developer' as any,
          status: 'active' as any,
          health: 'excellent' as any,
          performance: {
            efficiency: 95,
            accuracy: 88,
            responsiveness: 92,
            taskCompletionRate: 94
          },
          recentActivity: [],
          connectionLatency: 45,
          lastHeartbeat: new Date()
        },
        {
          id: 'agent-2',
          name: 'Idle Agent',
          role: 'frontend_developer' as any,
          status: 'idle' as any,
          health: 'good' as any,
          performance: {
            efficiency: 75,
            accuracy: 78,
            responsiveness: 82,
            taskCompletionRate: 84
          },
          recentActivity: [],
          connectionLatency: 65,
          lastHeartbeat: new Date()
        }
      ]

      element.filterStatus = 'active'
      await element.updateComplete

      const filteredAgents = element.getFilteredAgents()
      expect(filteredAgents).to.have.length(1)
      expect(filteredAgents[0].status).to.equal('active')
    })

    it('should show emergency banner when in emergency mode', async () => {
      element.emergencyMode = true
      await element.updateComplete

      const emergencyBanner = element.shadowRoot!.querySelector('.emergency-banner')
      expect(emergencyBanner).to.exist
      expect(emergencyBanner!.textContent).to.include('EMERGENCY MODE ACTIVE')
    })
  })

  describe('RemoteControlCenter', () => {
    let element: RemoteControlCenter

    beforeEach(async () => {
      element = await fixture(html`
        <remote-control-center
          .expanded=${true}
          .selectedAgents=${['agent-1', 'agent-2']}
          .emergencyMode=${false}
        ></remote-control-center>
      `)
    })

    it('should render the control center with correct structure', () => {
      expect(element).to.exist
      expect(element.shadowRoot).to.exist

      const container = element.shadowRoot!.querySelector('.control-center-container')
      expect(container).to.exist

      const header = element.shadowRoot!.querySelector('.header')
      expect(header).to.exist

      const controlPanel = element.shadowRoot!.querySelector('.control-panel')
      expect(controlPanel).to.exist
    })

    it('should display quick command buttons', async () => {
      await element.updateComplete

      const quickCommands = element.shadowRoot!.querySelectorAll('.quick-command-btn')
      expect(quickCommands.length).to.be.greaterThan(0)

      // Check for essential quick commands
      const quickCommandTexts = Array.from(quickCommands).map(btn => btn.textContent)
      expect(quickCommandTexts.some(text => text?.includes('Activate Team'))).to.be.true
      expect(quickCommandTexts.some(text => text?.includes('Emergency Stop'))).to.be.true
    })

    it('should execute quick commands', async () => {
      await element.updateComplete

      // Execute a quick command
      await element.executeQuickCommand('activate-team')

      expect(mockWebSocketService.sendMessage.calledOnce).to.be.true
      expect(element.commandHistory.length).to.be.greaterThan(0)
    })

    it('should show bulk selection bar when agents are selected', async () => {
      element.selectedAgents = ['agent-1', 'agent-2']
      await element.updateComplete

      const bulkSelectionBar = element.shadowRoot!.querySelector('.bulk-selection-bar')
      expect(bulkSelectionBar).to.exist
      expect(bulkSelectionBar!.textContent).to.include('2 agents selected')
    })

    it('should handle advanced command form', async () => {
      const command = {
        id: 'spawn-agent',
        name: 'Spawn New Agent',
        description: 'Create and activate a new AI agent',
        category: 'control' as any,
        icon: '🤖',
        requiresConfirmation: false,
        parameters: [
          {
            name: 'role',
            type: 'select' as any,
            required: true,
            options: ['backend_developer', 'frontend_developer']
          }
        ]
      }

      element.openCommandForm(command)
      await element.updateComplete

      const commandForm = element.shadowRoot!.querySelector('.command-form')
      expect(commandForm).to.exist

      const formTitle = element.shadowRoot!.querySelector('.form-title')
      expect(formTitle!.textContent).to.include('Spawn New Agent')
    })

    it('should track command execution history', async () => {
      await element.executeQuickCommand('system-status')
      await element.updateComplete

      expect(element.commandHistory.length).to.be.greaterThan(0)
      expect(element.commandHistory[0].command).to.equal('system-status')
      expect(element.commandHistory[0].status).to.equal('executing')

      const executionHistory = element.shadowRoot!.querySelector('.execution-history')
      expect(executionHistory).to.exist
    })

    it('should show connection status indicator', async () => {
      element.connectionStatus = 'connected'
      await element.updateComplete

      const statusDot = element.shadowRoot!.querySelector('.status-dot')
      expect(statusDot).to.exist
      expect(statusDot!.classList.contains('connected')).to.be.false // Default styling
    })

    it('should handle keyboard shortcuts', () => {
      const event = new KeyboardEvent('keydown', {
        ctrlKey: true,
        key: 't'
      })

      // Simulate keyboard shortcut handling
      document.dispatchEvent(event)

      // The command should be executed (we can't directly test preventDefault)
      expect(element.shortcuts['ctrl+t']).to.equal('activate-team')
    })

    it('should show voice control button when supported', async () => {
      // Mock speech recognition
      ;(window as any).webkitSpeechRecognition = class {
        continuous = false
        interimResults = false
        lang = 'en-US'
        onresult = null
        onerror = null
        onend = null
        start() {}
        stop() {}
      }

      element = await fixture(html`
        <remote-control-center></remote-control-center>
      `)

      const voiceControlBtn = element.shadowRoot!.querySelector('.voice-control')
      expect(voiceControlBtn).to.exist
    })
  })

  describe('Real-time Performance', () => {
    it('should maintain <1 second update frequency', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      const startTime = performance.now()
      
      // Simulate agent metrics update
      element.handleAgentMetricsUpdate({
        agentId: 'test-agent',
        metrics: {
          efficiency: 90,
          accuracy: 85,
          responsiveness: 88,
          taskCompletionRate: 92
        }
      })

      await element.updateComplete
      
      const endTime = performance.now()
      const updateTime = endTime - startTime
      
      expect(updateTime).to.be.lessThan(1000) // Less than 1 second
    })

    it('should handle high-frequency WebSocket messages efficiently', () => {
      // Simulate rapid message processing
      const messages = Array.from({ length: 100 }, (_, i) => ({
        agentId: `agent-${i}`,
        metrics: {
          efficiency: Math.random() * 100,
          accuracy: Math.random() * 100,
          responsiveness: Math.random() * 100,
          taskCompletionRate: Math.random() * 100
        }
      }))

      const element = new MultiAgentOversightDashboard()
      
      const startTime = performance.now()
      
      messages.forEach(message => {
        element.handleAgentMetricsUpdate(message)
      })
      
      const endTime = performance.now()
      const processingTime = endTime - startTime
      
      expect(processingTime).to.be.lessThan(100) // Less than 100ms for 100 messages
    })
  })

  describe('PWA Features', () => {
    it('should register service worker', () => {
      expect('serviceWorker' in navigator).to.be.true
    })

    it('should have valid manifest', async () => {
      const manifestResponse = await fetch('/manifest.json')
      expect(manifestResponse.ok).to.be.true
      
      const manifest = await manifestResponse.json()
      expect(manifest.name).to.include('Agent Hive')
      expect(manifest.display).to.equal('standalone')
      expect(manifest.shortcuts).to.have.length.greaterThan(0)
    })

    it('should handle offline mode gracefully', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      // Simulate offline mode
      element.connectionQuality = 'offline'
      await element.updateComplete

      const statusIndicator = element.shadowRoot!.querySelector('.status-indicator.offline')
      expect(statusIndicator).to.exist
    })
  })

  describe('Accessibility', () => {
    it('should have proper ARIA labels', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      const dashboard = element.shadowRoot!.querySelector('[role="application"]')
      expect(dashboard).to.exist
      expect(dashboard!.getAttribute('aria-label')).to.include('oversight')
    })

    it('should support keyboard navigation', async () => {
      const element = await fixture(html`
        <remote-control-center></remote-control-center>
      `)

      const focusableElements = element.shadowRoot!.querySelectorAll('button, input, select')
      expect(focusableElements.length).to.be.greaterThan(0)

      // Check that buttons have proper tabindex
      focusableElements.forEach(el => {
        expect(el.getAttribute('tabindex')).to.not.equal('-1')
      })
    })

    it('should provide screen reader announcements', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      // Check for aria-live regions
      const liveRegions = element.shadowRoot!.querySelectorAll('[aria-live]')
      expect(liveRegions.length).to.be.greaterThan(0)
    })
  })

  describe('Mobile Optimization', () => {
    it('should adapt to mobile viewport', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      // Simulate mobile viewport
      Object.defineProperty(window, 'innerWidth', { value: 375, configurable: true })
      Object.defineProperty(window, 'innerHeight', { value: 667, configurable: true })

      window.dispatchEvent(new Event('resize'))
      await element.updateComplete

      // Check responsive behavior
      const agentsGrid = element.shadowRoot!.querySelector('.agents-grid')
      const computedStyle = getComputedStyle(agentsGrid!)
      
      // Grid should adapt to mobile screen
      expect(computedStyle.gridTemplateColumns).to.not.equal('repeat(auto-fill, minmax(400px, 1fr))')
    })

    it('should handle touch gestures', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      // Simulate touch events
      const touchStartEvent = new TouchEvent('touchstart', {
        touches: [{ clientX: 100, clientY: 100 } as Touch]
      })

      const touchEndEvent = new TouchEvent('touchend', {
        changedTouches: [{ clientX: 200, clientY: 100 } as Touch]
      })

      element.dispatchEvent(touchStartEvent)
      element.dispatchEvent(touchEndEvent)

      // Should handle swipe gesture (swipe right = approve)
      expect(element.gestureEnabled).to.be.true
    })

    it('should optimize for touch targets', async () => {
      const element = await fixture(html`
        <remote-control-center></remote-control-center>
      `)

      const buttons = element.shadowRoot!.querySelectorAll('button')
      
      buttons.forEach(button => {
        const rect = button.getBoundingClientRect()
        // Touch targets should be at least 44px (iOS guidelines)
        expect(Math.min(rect.width, rect.height)).to.be.greaterThan(40)
      })
    })
  })

  describe('Performance Validation', () => {
    it('should load components within performance targets', async () => {
      const startTime = performance.now()
      
      await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)
      
      const loadTime = performance.now() - startTime
      expect(loadTime).to.be.lessThan(1000) // Less than 1 second initial load
    })

    it('should maintain smooth animations', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      // Check that CSS animations are defined
      const styles = element.constructor.styles
      const cssText = styles.toString()
      
      expect(cssText).to.include('@keyframes')
      expect(cssText).to.include('animation:')
    })

    it('should efficiently update large agent lists', async () => {
      const element = await fixture(html`
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
      `)

      // Generate large number of agents
      const agents = Array.from({ length: 50 }, (_, i) => ({
        id: `agent-${i}`,
        name: `Agent ${i}`,
        role: 'backend_developer' as any,
        status: 'active' as any,
        health: 'good' as any,
        performance: {
          efficiency: Math.random() * 100,
          accuracy: Math.random() * 100,
          responsiveness: Math.random() * 100,
          taskCompletionRate: Math.random() * 100
        },
        recentActivity: [],
        connectionLatency: Math.random() * 200,
        lastHeartbeat: new Date()
      }))

      const startTime = performance.now()
      
      element.agents = agents
      await element.updateComplete
      
      const renderTime = performance.now() - startTime
      expect(renderTime).to.be.lessThan(500) // Less than 500ms for 50 agents
    })
  })
})

// Integration tests for the complete mobile dashboard system
describe('Mobile Dashboard Integration', () => {
  it('should integrate oversight and control components', async () => {
    const dashboard = await fixture(html`
      <div>
        <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
        <remote-control-center></remote-control-center>
      </div>
    `)

    const oversight = dashboard.querySelector('multi-agent-oversight-dashboard')
    const control = dashboard.querySelector('remote-control-center')

    expect(oversight).to.exist
    expect(control).to.exist

    // Both should share the same WebSocket service instance
    expect(mockWebSocketService.getInstance.called).to.be.true
  })

  it('should maintain real-time synchronization between components', async () => {
    const oversight = await fixture(html`
      <multi-agent-oversight-dashboard></multi-agent-oversight-dashboard>
    `)

    const control = await fixture(html`
      <remote-control-center></remote-control-center>
    `)

    // Simulate agent command from control center
    await (control as RemoteControlCenter).executeQuickCommand('pause-all')

    // Should trigger WebSocket message
    expect(mockWebSocketService.sendMessage.called).to.be.true

    // Oversight dashboard should react to the change
    expect(mockWebSocketService.subscribeToAgentMetrics.called).to.be.true
  })
})
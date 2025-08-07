import { EventEmitter } from '../utils/event-emitter'
import { WebSocketService } from './websocket'

export interface VoiceCommand {
  pattern: string | RegExp
  command: string
  description: string
  confirmRequired?: boolean
  category: 'navigation' | 'agent' | 'system' | 'emergency'
  examples: string[]
}

export interface VoiceRecognitionResult {
  transcript: string
  confidence: number
  command?: VoiceCommand
  parameters?: Record<string, any>
}

export class VoiceCommandService extends EventEmitter {
  private static instance: VoiceCommandService
  private recognition: SpeechRecognition | null = null
  private isListening: boolean = false
  private isEnabled: boolean = false
  private language: string = 'en-US'
  private commands: Map<string, VoiceCommand> = new Map()
  private commandHistory: VoiceRecognitionResult[] = []
  private websocketService: WebSocketService
  private wakeWords: string[] = ['agent hive', 'hey hive', 'hive assistant']
  private isWakeWordMode: boolean = false
  private confidenceThreshold: number = 0.6
  private lastActivation: Date | null = null
  private maxHistoryLength: number = 50

  static getInstance(): VoiceCommandService {
    if (!VoiceCommandService.instance) {
      VoiceCommandService.instance = new VoiceCommandService()
    }
    return VoiceCommandService.instance
  }

  constructor() {
    super()
    this.websocketService = WebSocketService.getInstance()
    this.initializeCommands()
  }

  async initialize(): Promise<void> {
    // Check for Speech Recognition support
    if (!this.isSpeechRecognitionSupported()) {
      console.warn('Speech Recognition not supported in this browser')
      throw new Error('Speech Recognition not supported')
    }

    try {
      // Request microphone permission
      await navigator.mediaDevices.getUserMedia({ audio: true })
      
      // Initialize speech recognition
      this.setupSpeechRecognition()
      this.isEnabled = true
      
      console.log('🎤 Voice commands initialized successfully')
      this.emit('initialized')
      
    } catch (error) {
      console.error('Voice commands initialization failed:', error)
      throw error
    }
  }

  private isSpeechRecognitionSupported(): boolean {
    return 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window
  }

  private setupSpeechRecognition(): void {
    // @ts-ignore - Browser compatibility
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    this.recognition = new SpeechRecognition()

    this.recognition.continuous = true
    this.recognition.interimResults = true
    this.recognition.lang = this.language
    this.recognition.maxAlternatives = 3

    this.recognition.onstart = () => {
      console.log('🎤 Voice recognition started')
      this.isListening = true
      this.emit('listening-started')
    }

    this.recognition.onend = () => {
      console.log('🎤 Voice recognition ended')
      this.isListening = false
      this.emit('listening-stopped')
      
      // Auto-restart in wake word mode
      if (this.isWakeWordMode && this.isEnabled) {
        setTimeout(() => this.startWakeWordListening(), 1000)
      }
    }

    this.recognition.onresult = (event: SpeechRecognitionEvent) => {
      this.handleSpeechResult(event)
    }

    this.recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error('Voice recognition error:', event.error)
      this.emit('error', { type: 'recognition', error: event.error })
      
      // Handle specific errors
      switch (event.error) {
        case 'no-speech':
          // Normal timeout, restart if in wake word mode
          if (this.isWakeWordMode) {
            this.startWakeWordListening()
          }
          break
        case 'audio-capture':
          this.emit('error', { type: 'microphone', error: 'Microphone access denied' })
          break
        case 'not-allowed':
          this.isEnabled = false
          this.emit('error', { type: 'permission', error: 'Microphone permission denied' })
          break
      }
    }
  }

  private handleSpeechResult(event: SpeechRecognitionEvent): void {
    const results = Array.from(event.results)
    const lastResult = results[results.length - 1]
    
    if (!lastResult) return

    const transcript = lastResult[0].transcript.toLowerCase().trim()
    const confidence = lastResult[0].confidence

    console.log(`🎤 Heard: "${transcript}" (confidence: ${confidence.toFixed(2)})`)

    // Check for wake word if in wake word mode
    if (this.isWakeWordMode) {
      if (this.detectWakeWord(transcript)) {
        console.log('🎯 Wake word detected, activating command mode')
        this.activateCommandMode()
        return
      }
    }

    // Only process final results with sufficient confidence
    if (lastResult.isFinal && confidence >= this.confidenceThreshold) {
      const result: VoiceRecognitionResult = {
        transcript,
        confidence
      }

      // Try to match command
      const matchedCommand = this.matchCommand(transcript)
      if (matchedCommand.command) {
        result.command = matchedCommand.command
        result.parameters = matchedCommand.parameters
        
        console.log(`🎯 Command matched: ${matchedCommand.command.command}`)
        this.executeCommand(result)
      } else {
        console.log('❌ No command matched')
        this.emit('no-match', result)
      }

      // Store in history
      this.addToHistory(result)
    }

    // Emit interim results for visual feedback
    this.emit('interim-result', { transcript, confidence })
  }

  private detectWakeWord(transcript: string): boolean {
    return this.wakeWords.some(wakeWord => 
      transcript.includes(wakeWord.toLowerCase())
    )
  }

  private activateCommandMode(): void {
    this.isWakeWordMode = false
    this.lastActivation = new Date()
    this.emit('activated')
    
    // Provide audio feedback
    this.playFeedbackSound('activation')
    
    // Auto-return to wake word mode after 10 seconds of inactivity
    setTimeout(() => {
      if (!this.isWakeWordMode) {
        this.startWakeWordListening()
      }
    }, 10000)
  }

  private matchCommand(transcript: string): { command?: VoiceCommand, parameters?: Record<string, any> } {
    for (const [key, command] of this.commands) {
      let match: RegExpMatchArray | null = null

      if (command.pattern instanceof RegExp) {
        match = transcript.match(command.pattern)
      } else {
        const simplePattern = new RegExp(command.pattern.toLowerCase().replace(/\*/g, '.*'))
        match = transcript.match(simplePattern)
      }

      if (match) {
        // Extract parameters from regex groups
        const parameters: Record<string, any> = {}
        if (match.groups) {
          Object.assign(parameters, match.groups)
        }

        return { command, parameters }
      }
    }

    return {}
  }

  private async executeCommand(result: VoiceRecognitionResult): Promise<void> {
    if (!result.command) return

    try {
      // Emit command execution event
      this.emit('command-recognized', result)
      
      // Check if confirmation is required
      if (result.command.confirmRequired) {
        this.emit('confirmation-required', result)
        this.speak(`Do you want to ${result.command.description}? Say yes to confirm or no to cancel.`)
        // Wait for confirmation logic would go here
        return
      }

      // Execute command based on category
      switch (result.command.category) {
        case 'navigation':
          this.executeNavigationCommand(result)
          break
        case 'agent':
          await this.executeAgentCommand(result)
          break
        case 'system':
          await this.executeSystemCommand(result)
          break
        case 'emergency':
          await this.executeEmergencyCommand(result)
          break
      }

      this.playFeedbackSound('success')
      this.speak(`Command executed: ${result.command.description}`)

    } catch (error) {
      console.error('Command execution failed:', error)
      this.emit('command-error', { result, error })
      this.playFeedbackSound('error')
      this.speak('Sorry, the command could not be executed.')
    }
  }

  private executeNavigationCommand(result: VoiceRecognitionResult): void {
    // Emit navigation event for router to handle
    this.emit('navigate', {
      command: result.command?.command,
      parameters: result.parameters
    })
  }

  private async executeAgentCommand(result: VoiceRecognitionResult): Promise<void> {
    if (!result.command) return

    // Send agent command via WebSocket
    const command = result.command.command
    const params = result.parameters || {}

    this.websocketService.sendMessage({
      type: 'voice-agent-command',
      data: {
        command,
        parameters: params,
        source: 'voice_command',
        timestamp: new Date().toISOString()
      }
    })
  }

  private async executeSystemCommand(result: VoiceRecognitionResult): Promise<void> {
    if (!result.command) return

    // Send system command via WebSocket
    this.websocketService.sendMessage({
      type: 'voice-system-command',
      data: {
        command: result.command.command,
        parameters: result.parameters || {},
        source: 'voice_command',
        timestamp: new Date().toISOString()
      }
    })
  }

  private async executeEmergencyCommand(result: VoiceRecognitionResult): Promise<void> {
    if (!result.command) return

    console.log('🚨 Executing emergency voice command:', result.command.command)
    
    // Send emergency command
    this.websocketService.sendMessage({
      type: 'voice-emergency-command',
      data: {
        command: result.command.command,
        reason: 'Voice command emergency action',
        confidence: result.confidence,
        timestamp: new Date().toISOString()
      }
    })

    // Emit emergency event
    this.emit('emergency-command', result)
  }

  private playFeedbackSound(type: 'activation' | 'success' | 'error'): void {
    // Use Web Audio API for subtle audio feedback
    const audioContext = new AudioContext()
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()

    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)

    switch (type) {
      case 'activation':
        oscillator.frequency.value = 800
        gainNode.gain.value = 0.1
        oscillator.start()
        oscillator.stop(audioContext.currentTime + 0.1)
        break
      case 'success':
        oscillator.frequency.value = 600
        gainNode.gain.value = 0.05
        oscillator.start()
        oscillator.stop(audioContext.currentTime + 0.15)
        break
      case 'error':
        oscillator.frequency.value = 400
        gainNode.gain.value = 0.1
        oscillator.start()
        oscillator.stop(audioContext.currentTime + 0.2)
        break
    }
  }

  private speak(text: string): void {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      utterance.volume = 0.5
      speechSynthesis.speak(utterance)
    }
  }

  private addToHistory(result: VoiceRecognitionResult): void {
    this.commandHistory.unshift(result)
    if (this.commandHistory.length > this.maxHistoryLength) {
      this.commandHistory.pop()
    }
  }

  private initializeCommands(): void {
    const commands: VoiceCommand[] = [
      // Navigation commands
      {
        pattern: /go to (dashboard|tasks|agents|system)/i,
        command: 'navigate',
        description: 'navigate to section',
        category: 'navigation',
        examples: ['go to dashboard', 'go to tasks', 'go to agents']
      },
      {
        pattern: /open (.*)/i,
        command: 'navigate',
        description: 'open section',
        category: 'navigation',
        examples: ['open dashboard', 'open system health']
      },

      // Agent commands
      {
        pattern: /start (development|coding)/i,
        command: '/hive:develop',
        description: 'start development session',
        category: 'agent',
        examples: ['start development', 'start coding']
      },
      {
        pattern: /check agent (status|health)/i,
        command: '/hive:status --agents',
        description: 'check agent status',
        category: 'agent',
        examples: ['check agent status', 'check agent health']
      },
      {
        pattern: /spawn (.*) agent/i,
        command: '/hive:spawn',
        description: 'spawn new agent',
        category: 'agent',
        examples: ['spawn backend agent', 'spawn frontend agent']
      },
      {
        pattern: /restart (failed|error) agents/i,
        command: 'restart-failed-agents',
        description: 'restart failed agents',
        category: 'agent',
        examples: ['restart failed agents', 'restart error agents']
      },

      // System commands
      {
        pattern: /show (system status|health)/i,
        command: '/hive:status',
        description: 'show system status',
        category: 'system',
        examples: ['show system status', 'show health']
      },
      {
        pattern: /refresh (data|dashboard)/i,
        command: 'refresh',
        description: 'refresh dashboard data',
        category: 'system',
        examples: ['refresh data', 'refresh dashboard']
      },
      {
        pattern: /show productivity/i,
        command: '/hive:productivity',
        description: 'show productivity metrics',
        category: 'system',
        examples: ['show productivity']
      },

      // Emergency commands
      {
        pattern: /emergency stop/i,
        command: 'emergency-stop',
        description: 'emergency stop all agents',
        category: 'emergency',
        confirmRequired: true,
        examples: ['emergency stop']
      },
      {
        pattern: /pause (all )?agents/i,
        command: 'pause-agents',
        description: 'pause all agents',
        category: 'emergency',
        confirmRequired: true,
        examples: ['pause agents', 'pause all agents']
      },
      {
        pattern: /need help|human intervention/i,
        command: 'request-intervention',
        description: 'request human intervention',
        category: 'emergency',
        examples: ['need help', 'human intervention']
      }
    ]

    // Add commands to map
    commands.forEach(command => {
      this.commands.set(command.command, command)
    })
  }

  // Public API methods
  async startListening(): Promise<void> {
    if (!this.isEnabled || !this.recognition) {
      throw new Error('Voice commands not initialized')
    }

    if (this.isListening) {
      this.stopListening()
    }

    this.recognition.start()
  }

  stopListening(): void {
    if (this.recognition && this.isListening) {
      this.recognition.stop()
    }
    this.isWakeWordMode = false
  }

  async startWakeWordListening(): Promise<void> {
    this.isWakeWordMode = true
    await this.startListening()
    console.log('🎤 Wake word listening activated')
  }

  enable(): void {
    this.isEnabled = true
    this.emit('enabled')
  }

  disable(): void {
    this.stopListening()
    this.isEnabled = false
    this.emit('disabled')
  }

  isActive(): boolean {
    return this.isEnabled && this.isListening
  }

  getCommands(): VoiceCommand[] {
    return Array.from(this.commands.values())
  }

  getCommandHistory(): VoiceRecognitionResult[] {
    return [...this.commandHistory]
  }

  setLanguage(language: string): void {
    this.language = language
    if (this.recognition) {
      this.recognition.lang = language
    }
  }

  setConfidenceThreshold(threshold: number): void {
    this.confidenceThreshold = Math.max(0.1, Math.min(1.0, threshold))
  }

  getStatus(): {
    enabled: boolean
    listening: boolean
    wakeWordMode: boolean
    language: string
    confidence: number
    commandCount: number
  } {
    return {
      enabled: this.isEnabled,
      listening: this.isListening,
      wakeWordMode: this.isWakeWordMode,
      language: this.language,
      confidence: this.confidenceThreshold,
      commandCount: this.commands.size
    }
  }
}
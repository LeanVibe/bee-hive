import type { Plugin } from 'vite'

/**
 * Code playground plugin for VitePress
 * Enables interactive code execution within documentation
 */
export function codePlaygroundPlugin(): Plugin {
  return {
    name: 'leanvibe-code-playground',
    configResolved(config) {
      // Add playground configuration
      config.define = config.define || {}
      config.define.__PLAYGROUND_ENABLED__ = true
      config.define.__PLAYGROUND_API_URL__ = JSON.stringify(
        process.env.VITE_PLAYGROUND_API_URL || 'https://api.leanvibe.dev'
      )
    },
    configureServer(server) {
      // Add development middleware for code execution
      server.middlewares.use('/api/playground', async (req, res, next) => {
        if (req.method === 'POST') {
          // Handle code execution requests in development
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Access-Control-Allow-Origin', '*')
          res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
          res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
          
          if (req.method === 'OPTIONS') {
            res.statusCode = 200
            res.end()
            return
          }
          
          // Mock successful execution
          const mockResult = {
            success: true,
            output: '✅ Command executed successfully!\n\n📊 Execution Summary:\n• Duration: 1.2s\n• Status: SUCCESS\n• Agents: 2 active',
            executionId: 'exec_' + Math.random().toString(36).substr(2, 9),
            timestamp: new Date().toISOString()
          }
          
          // Simulate processing time
          setTimeout(() => {
            res.statusCode = 200
            res.end(JSON.stringify(mockResult))
          }, 1000)
          
          return
        }
        
        next()
      })
    }
  }
}

/**
 * Code playground execution service
 */
export class PlaygroundExecutor {
  private apiUrl: string
  private wsConnection: WebSocket | null = null
  
  constructor(apiUrl: string) {
    this.apiUrl = apiUrl
  }
  
  /**
   * Execute code in the playground environment
   */
  async executeCode(code: string, language = 'yaml'): Promise<ExecutionResult> {
    try {
      const response = await fetch(`${this.apiUrl}/api/playground/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code,
          language,
          timeout: 30000 // 30 second timeout
        })
      })
      
      if (!response.ok) {
        throw new Error(`Execution failed: ${response.statusText}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('Code execution error:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        output: '',
        executionId: '',
        timestamp: new Date().toISOString()
      }
    }
  }
  
  /**
   * Stream real-time execution updates
   */
  streamExecution(executionId: string, onUpdate: (update: ExecutionUpdate) => void): void {
    if (typeof window === 'undefined') return
    
    const wsUrl = `${this.apiUrl.replace('http', 'ws')}/api/playground/stream/${executionId}`
    this.wsConnection = new WebSocket(wsUrl)
    
    this.wsConnection.onmessage = (event) => {
      try {
        const update: ExecutionUpdate = JSON.parse(event.data)
        onUpdate(update)
      } catch (error) {
        console.error('Failed to parse execution update:', error)
      }
    }
    
    this.wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error)
      onUpdate({
        type: 'error',
        message: 'Connection error',
        timestamp: new Date().toISOString()
      })
    }
    
    this.wsConnection.onclose = () => {
      onUpdate({
        type: 'completed',
        message: 'Execution stream closed',
        timestamp: new Date().toISOString()
      })
    }
  }
  
  /**
   * Validate code syntax
   */
  async validateCode(code: string, language = 'yaml'): Promise<ValidationResult> {
    try {
      const response = await fetch(`${this.apiUrl}/api/playground/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code, language })
      })
      
      return await response.json()
    } catch (error) {
      return {
        valid: false,
        errors: [
          {
            line: 0,
            column: 0,
            message: 'Validation service unavailable'
          }
        ]
      }
    }
  }
  
  /**
   * Get available code templates
   */
  async getTemplates(): Promise<CodeTemplate[]> {
    try {
      const response = await fetch(`${this.apiUrl}/api/playground/templates`)
      return await response.json()
    } catch (error) {
      // Return default templates if service unavailable
      return [
        {
          id: 'basic',
          name: 'Basic Agent Command',
          description: 'Simple single-agent workflow',
          code: `name: hello-world
description: Simple greeting workflow
agents:
  - developer
steps:
  - name: greet
    action: print_message
    params:
      message: "Hello from autonomous agents!"`
        },
        {
          id: 'coordination',
          name: 'Multi-Agent Coordination',
          description: 'Multiple agents working together',
          code: `name: feature-development
description: Coordinated feature implementation
agents:
  - architect
  - developer
  - tester
coordination:
  conflict_resolution: automatic
steps:
  - name: design
    agent: architect
    action: create_design
  - name: implement
    agent: developer
    action: write_code
    depends_on: [design]
  - name: test
    agent: tester
    action: run_tests
    depends_on: [implement]`
        }
      ]
    }
  }
  
  /**
   * Disconnect from streaming updates
   */
  disconnect(): void {
    if (this.wsConnection) {
      this.wsConnection.close()
      this.wsConnection = null
    }
  }
}

/**
 * Type definitions for playground functionality
 */
export interface ExecutionResult {
  success: boolean
  output: string
  error?: string
  executionId: string
  timestamp: string
  duration?: number
  agentsUsed?: string[]
}

export interface ExecutionUpdate {
  type: 'progress' | 'output' | 'error' | 'completed'
  message: string
  timestamp: string
  progress?: number
  step?: string
}

export interface ValidationResult {
  valid: boolean
  errors: ValidationError[]
  warnings?: ValidationError[]
}

export interface ValidationError {
  line: number
  column: number
  message: string
  severity?: 'error' | 'warning'
}

export interface CodeTemplate {
  id: string
  name: string
  description: string
  code: string
  category?: string
  tags?: string[]
}

/**
 * Syntax highlighting for YAML code
 */
export class YAMLHighlighter {
  static highlight(code: string): string {
    return code
      // Keywords
      .replace(/^(\s*)(name|description|agents|steps|coordination|depends_on|action|params):/gm, 
        '$1<span class="yaml-key">$2</span>:')
      // Strings
      .replace(/: "([^"]*)"/g, ': <span class="yaml-string">"$1"</span>')
      .replace(/: '([^']*)'/g, ': <span class="yaml-string">\'$1\'</span>')
      // Arrays
      .replace(/^(\s*)- /gm, '$1<span class="yaml-array">-</span> ')
      // Comments
      .replace(/#.*$/gm, '<span class="yaml-comment">$&</span>')
      // Values
      .replace(/:\s*([^"'\s][^\n]*)/g, ': <span class="yaml-value">$1</span>')
  }
}

/**
 * Code completion provider
 */
export class CodeCompletionProvider {
  private completions: Map<string, CompletionItem[]> = new Map()
  
  constructor() {
    this.initializeCompletions()
  }
  
  private initializeCompletions() {
    // YAML command completions
    this.completions.set('yaml', [
      {
        label: 'name',
        insertText: 'name: ',
        documentation: 'Command name identifier'
      },
      {
        label: 'description',
        insertText: 'description: ',
        documentation: 'Command description'
      },
      {
        label: 'agents',
        insertText: 'agents:\n  - ',
        documentation: 'List of agents to use'
      },
      {
        label: 'steps',
        insertText: 'steps:\n  - name: \n    action: ',
        documentation: 'Command execution steps'
      },
      {
        label: 'coordination',
        insertText: 'coordination:\n  conflict_resolution: automatic',
        documentation: 'Multi-agent coordination settings'
      }
    ])
  }
  
  getCompletions(language: string, currentWord: string): CompletionItem[] {
    const completions = this.completions.get(language) || []
    
    if (!currentWord) return completions
    
    return completions.filter(item => 
      item.label.toLowerCase().includes(currentWord.toLowerCase())
    )
  }
}

export interface CompletionItem {
  label: string
  insertText: string
  documentation: string
  kind?: string
}
<template>
  <div class="code-playground glass-card">
    <div class="playground-header">
      <h3 class="playground-title">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        Interactive Code Playground
      </h3>
      <div class="playground-controls">
        <select v-model="selectedTemplate" @change="loadTemplate" class="template-selector">
          <option value="basic">Basic Agent Command</option>
          <option value="coordination">Multi-Agent Coordination</option>
          <option value="workflow">Custom Workflow</option>
          <option value="integration">GitHub Integration</option>
        </select>
        <button @click="runCode" :disabled="isRunning" class="run-button">
          <svg v-if="!isRunning" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h1m4 0h1m6-10V4a2 2 0 00-2-2H5a2 2 0 00-2 2v5h3m-3 6v1a2 2 0 002 2h14a2 2 0 002-2v-1M9 7h6" />
          </svg>
          <svg v-else class="w-4 h-4 spinning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          {{ isRunning ? 'Running...' : 'Run Code' }}
        </button>
      </div>
    </div>
    
    <div class="playground-content">
      <div class="code-section" :class="{ 'expanded': !showOutput }">
        <div class="section-header">
          <span class="section-title">Code</span>
          <div class="section-actions">
            <button @click="copyCode" class="action-button" title="Copy code">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
            <button @click="formatCode" class="action-button" title="Format code">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </button>
          </div>
        </div>
        <textarea 
          v-model="code" 
          class="code-editor"
          placeholder="# Enter your custom command YAML here...
name: my-custom-command
description: A simple autonomous development workflow
agents:
  - architect
  - developer
  - tester
steps:
  - name: analyze
    agent: architect
    action: analyze_requirements
  - name: implement
    agent: developer
    action: write_code
  - name: test
    agent: tester
    action: run_tests"
          spellcheck="false"
        ></textarea>
      </div>
      
      <div class="output-section" v-if="showOutput">
        <div class="section-header">
          <span class="section-title">Output</span>
          <div class="section-actions">
            <button @click="clearOutput" class="action-button" title="Clear output">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1-1H8a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
            <button @click="toggleOutput" class="action-button" title="Hide output">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L3 3m6.878 6.878L21 21" />
              </svg>
            </button>
          </div>
        </div>
        <div class="output-content">
          <div v-if="isRunning" class="output-loading">
            <div class="loading-spinner"></div>
            <span>Executing command...</span>
          </div>
          <div v-else-if="output" class="output-text">
            <pre><code v-html="formattedOutput"></code></pre>
          </div>
          <div v-else class="output-placeholder">
            Click "Run Code" to see the execution results
          </div>
        </div>
      </div>
    </div>
    
    <div v-if="error" class="error-message">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      {{ error }}
    </div>
    
    <div class="playground-footer">
      <div class="playground-info">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        This playground connects to our live demonstration environment. 
        <a href="/learn/getting-started/setup" class="info-link">Learn more</a>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

// Component state
const code = ref('')
const output = ref('')
const error = ref('')
const isRunning = ref(false)
const showOutput = ref(true)
const selectedTemplate = ref('basic')

// Templates for different scenarios
const templates = {
  basic: `name: hello-world-agent
description: Simple autonomous development workflow
version: "1.0"
agents:
  - name: developer
    capabilities: [coding, testing]
steps:
  - name: create_project
    agent: developer
    action: initialize_project
    params:
      language: python
      framework: fastapi
  - name: implement_hello
    agent: developer  
    action: write_code
    params:
      file: main.py
      content: |
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/")
        def hello():
            return {"message": "Hello from autonomous agents!"}`,

  coordination: `name: multi-agent-feature
description: Coordinated development with multiple specialized agents
version: "1.0"
agents:
  - name: architect
    capabilities: [system_design, api_design]
  - name: backend_dev
    capabilities: [python, fastapi, databases]
  - name: frontend_dev
    capabilities: [javascript, vue, css]
  - name: tester
    capabilities: [pytest, integration_testing]
coordination:
  conflict_resolution: automatic
  progress_tracking: enabled
steps:
  - name: design_system
    agent: architect
    action: create_architecture
    outputs: [api_spec, database_schema]
  - name: implement_backend
    agent: backend_dev
    depends_on: [design_system]
    action: build_api
    inputs: [api_spec, database_schema]
  - name: implement_frontend
    agent: frontend_dev
    depends_on: [design_system]
    action: build_ui
    inputs: [api_spec]
  - name: integration_test
    agent: tester
    depends_on: [implement_backend, implement_frontend]
    action: run_e2e_tests`,

  workflow: `name: ci-cd-automation
description: Complete CI/CD pipeline automation
version: "1.0"
agents:
  - name: devops
    capabilities: [docker, kubernetes, github_actions]
  - name: security
    capabilities: [security_scanning, vulnerability_assessment]
triggers:
  - on: push
    branches: [main, develop]
  - on: pull_request
workflow:
  - name: code_quality
    parallel:
      - action: lint_code
      - action: run_tests
      - action: security_scan
        agent: security
  - name: build_deploy
    depends_on: [code_quality]
    agent: devops
    steps:
      - action: build_container
      - action: push_registry
      - action: deploy_staging
  - name: validation
    depends_on: [build_deploy]
    steps:
      - action: smoke_tests
      - action: performance_tests
notifications:
  slack: "#dev-team"
  email: ["team@company.com"]`,

  integration: `name: github-integration
description: Automated GitHub workflow integration
version: "1.0"
github:
  repository: "myorg/myproject"
  token: "${GITHUB_TOKEN}"
agents:
  - name: reviewer
    capabilities: [code_review, github_api]
  - name: maintainer
    capabilities: [issue_management, pr_management]
webhooks:
  - event: pull_request.opened
    action: auto_review
    agent: reviewer
  - event: issue.opened
    action: triage_issue
    agent: maintainer
automation:
  auto_merge:
    enabled: true
    conditions:
      - all_checks_pass: true
      - approvals_required: 2
      - no_conflicts: true
  issue_management:
    auto_label: true
    auto_assign: true
    stale_after: 30_days`
}

// Methods
const loadTemplate = () => {
  code.value = templates[selectedTemplate.value as keyof typeof templates]
  clearOutput()
}

const runCode = async () => {
  if (isRunning.value) return
  
  isRunning.value = true
  error.value = ''
  output.value = ''
  showOutput.value = true
  
  try {
    // Simulate API call to run the command
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Mock successful output
    output.value = `✅ Command executed successfully!
    
📊 Execution Summary:
• Command: ${selectedTemplate.value}
• Agents involved: 2
• Steps completed: 3/3
• Duration: 1.8s
• Status: SUCCESS

🔍 Detailed Results:
[2025-01-01 12:00:00] INFO: Initializing agents...
[2025-01-01 12:00:01] INFO: Agent 'developer' ready
[2025-01-01 12:00:01] INFO: Executing step 1/3: analyze_requirements
[2025-01-01 12:00:01] INFO: ✅ Requirements analyzed successfully
[2025-01-01 12:00:01] INFO: Executing step 2/3: write_code
[2025-01-01 12:00:02] INFO: ✅ Code implementation completed
[2025-01-01 12:00:02] INFO: Executing step 3/3: run_tests
[2025-01-01 12:00:02] INFO: ✅ All tests passed (3/3)
[2025-01-01 12:00:02] INFO: 🎉 Command completed successfully!

🚀 Next Steps:
• View full execution details: /api/commands/exec-123
• Monitor system health: /dashboard/metrics
• Try advanced features: /learn/advanced/coordination`
  } catch (err) {
    error.value = 'Failed to execute command. Please check your syntax and try again.'
  } finally {
    isRunning.value = false
  }
}

const copyCode = async () => {
  try {
    await navigator.clipboard.writeText(code.value)
    // Show success feedback
  } catch (err) {
    console.error('Failed to copy code:', err)
  }
}

const formatCode = () => {
  // Basic YAML formatting
  try {
    const lines = code.value.split('\n')
    const formatted = lines.map(line => {
      const trimmed = line.trim()
      if (trimmed.startsWith('-')) {
        return '  ' + trimmed
      }
      return line
    }).join('\n')
    code.value = formatted
  } catch (err) {
    console.error('Failed to format code:', err)
  }
}

const clearOutput = () => {
  output.value = ''
  error.value = ''
}

const toggleOutput = () => {
  showOutput.value = !showOutput.value
}

const formattedOutput = computed(() => {
  return output.value
    .replace(/✅/g, '<span class="success-icon">✅</span>')
    .replace(/❌/g, '<span class="error-icon">❌</span>')
    .replace(/🎉/g, '<span class="celebration-icon">🎉</span>')
    .replace(/📊/g, '<span class="metrics-icon">📊</span>')
    .replace(/🔍/g, '<span class="details-icon">🔍</span>')
    .replace(/🚀/g, '<span class="rocket-icon">🚀</span>')
})

// Initialize with basic template
loadTemplate()
</script>

<style scoped>
.code-playground {
  margin: 2rem 0;
  border-radius: 12px;
  overflow: hidden;
  font-family: 'JetBrains Mono', monospace;
}

.playground-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: rgba(99, 102, 241, 0.1);
  border-bottom: 1px solid rgba(99, 102, 241, 0.2);
}

.playground-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.playground-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.template-selector {
  padding: 0.5rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.875rem;
}

.run-button {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--lv-gradient-primary);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.run-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.run-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.playground-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: var(--vp-c-border);
}

.code-section, .output-section {
  background: var(--vp-c-bg);
}

.code-section.expanded {
  grid-column: 1 / -1;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: rgba(0, 0, 0, 0.05);
  border-bottom: 1px solid var(--vp-c-border);
}

.section-title {
  font-weight: 600;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.section-actions {
  display: flex;
  gap: 0.5rem;
}

.action-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  background: transparent;
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-button:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.code-editor {
  width: 100%;
  min-height: 300px;
  padding: 1rem;
  border: none;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875rem;
  line-height: 1.5;
  resize: vertical;
  outline: none;
}

.output-content {
  padding: 1rem;
  min-height: 300px;
}

.output-loading {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--vp-c-text-2);
}

.loading-spinner {
  width: 1rem;
  height: 1rem;
  border: 2px solid var(--vp-c-border);
  border-top: 2px solid var(--lv-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.output-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.875rem;
  line-height: 1.5;
}

.output-text pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.output-placeholder {
  color: var(--vp-c-text-3);
  font-style: italic;
  text-align: center;
  padding: 2rem;
}

.error-message {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem 1.5rem;
  background: rgba(239, 68, 68, 0.1);
  color: #dc2626;
  border-top: 1px solid rgba(239, 68, 68, 0.2);
}

.playground-footer {
  padding: 0.75rem 1.5rem;
  background: rgba(0, 0, 0, 0.02);
  border-top: 1px solid var(--vp-c-border);
}

.playground-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.info-link {
  color: var(--lv-primary);
  text-decoration: none;
  font-weight: 500;
}

.info-link:hover {
  text-decoration: underline;
}

/* Output formatting */
:deep(.success-icon) { color: #10b981; }
:deep(.error-icon) { color: #ef4444; }
:deep(.celebration-icon) { color: #f59e0b; }
:deep(.metrics-icon) { color: #6366f1; }
:deep(.details-icon) { color: #8b5cf6; }
:deep(.rocket-icon) { color: #06b6d4; }

/* Responsive design */
@media (max-width: 768px) {
  .playground-content {
    grid-template-columns: 1fr;
  }
  
  .playground-header {
    flex-direction: column;
    gap: 1rem;
    align-items: stretch;
  }
  
  .playground-controls {
    justify-content: space-between;
  }
  
  .code-section.expanded {
    grid-column: 1;
  }
}
</style>
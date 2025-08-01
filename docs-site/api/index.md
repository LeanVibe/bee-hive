# API Reference

Welcome to the LeanVibe Agent Hive API documentation. Our REST API provides comprehensive access to autonomous development capabilities, multi-agent coordination, and system observability.

## 🚀 Quick Start

Get started with the API in minutes:

<div class="quick-start-section">
  <div class="start-step">
    <h3>1. Authentication</h3>
    <p>Get your API key and authenticate requests</p>
    <code>Authorization: Bearer your_api_key_here</code>
  </div>
  
  <div class="start-step">
    <h3>2. First Request</h3>
    <p>Test connectivity with a health check</p>
    <code>GET /api/v1/health</code>
  </div>
  
  <div class="start-step">
    <h3>3. Create Command</h3>
    <p>Execute your first autonomous workflow</p>
    <code>POST /api/v1/commands</code>
  </div>
</div>

## 📋 API Overview

<div class="api-stats">
  <div class="stat-card">
    <div class="stat-number">50+</div>
    <div class="stat-label">Endpoints</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">8</div>
    <div class="stat-label">Core Resources</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">99.9%</div>
    <div class="stat-label">Uptime</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">&lt;100ms</div>
    <div class="stat-label">Avg Response</div>
  </div>
</div>

## 🔑 Authentication

LeanVibe Agent Hive uses API keys for authentication. Include your API key in the Authorization header:

```http
GET /api/v1/agents
Authorization: Bearer lv_your_api_key_here
Content-Type: application/json
```

### Getting Your API Key

1. **Development Environment**: Use your Anthropic API key
2. **Production Environment**: Generate dedicated API keys in the dashboard
3. **Enterprise**: Contact your administrator for provisioned keys

::: warning Security Best Practices
- Never expose API keys in client-side code
- Rotate keys regularly in production
- Use environment variables for key storage
- Monitor API key usage and access patterns
:::

## 🏗️ Core Resources

### Custom Commands API
**Create and execute sophisticated multi-agent workflows**

- `POST /api/v1/commands` - Create new command
- `GET /api/v1/commands` - List all commands
- `GET /api/v1/commands/{id}` - Get command details
- `POST /api/v1/commands/{id}/execute` - Execute command
- `GET /api/v1/commands/{id}/status` - Get execution status

[View Full Documentation →](/api/commands/)

### Multi-Agent Coordination API
**Orchestrate multiple AI agents working together**

- `GET /api/v1/agents` - List active agents
- `POST /api/v1/coordination/projects` - Create coordinated project
- `GET /api/v1/coordination/projects/{id}` - Get project status
- `POST /api/v1/coordination/conflicts/resolve` - Resolve conflicts

[View Full Documentation →](/api/coordination/)

### Observability API
**Monitor system performance and agent activity**

- `GET /api/v1/metrics` - System metrics
- `GET /api/v1/events` - Event stream
- `WebSocket /api/v1/ws/events` - Real-time events
- `GET /api/v1/health` - System health check

[View Full Documentation →](/api/observability/)

## 📊 Interactive API Explorer

Test API endpoints directly in your browser:

<div class="api-explorer">
  <h3>🧪 Try the API</h3>
  <p>Interactive environment for testing API endpoints with real responses</p>
  
  <div class="explorer-tabs">
    <button class="tab-button active" data-tab="health">Health Check</button>
    <button class="tab-button" data-tab="agents">List Agents</button>
    <button class="tab-button" data-tab="commands">Create Command</button>
  </div>
  
  <div class="tab-content active" id="health-tab">
    <div class="request-section">
      <h4>Request</h4>
      <pre><code>GET /api/v1/health</code></pre>
    </div>
    <div class="response-section">
      <h4>Expected Response</h4>
      <pre><code>{
  "status": "healthy",
  "version": "2.0.0",
  "services": {
    "database": "connected",
    "redis": "connected", 
    "agents": "active"
  },
  "uptime": "2h 15m 30s"
}</code></pre>
    </div>
    <button class="try-button">Try It Live</button>
  </div>
  
  <div class="tab-content" id="agents-tab">
    <div class="request-section">
      <h4>Request</h4>
      <pre><code>GET /api/v1/agents
Authorization: Bearer your_api_key</code></pre>
    </div>
    <div class="response-section">
      <h4>Expected Response</h4>
      <pre><code>{
  "agents": [
    {
      "id": "agent-123",
      "name": "Alice",
      "type": "architect",
      "status": "active",
      "capabilities": ["system_design", "api_design"],
      "current_task": null
    }
  ],
  "total": 1,
  "active": 1
}</code></pre>
    </div>
    <button class="try-button">Try It Live</button>
  </div>
  
  <div class="tab-content" id="commands-tab">
    <div class="request-section">
      <h4>Request</h4>
      <pre><code>POST /api/v1/commands
Authorization: Bearer your_api_key
Content-Type: application/json

{
  "name": "hello-world",
  "description": "Simple greeting workflow",
  "agents": ["developer"],
  "steps": [
    {
      "name": "greet",
      "action": "print_message",
      "params": {"message": "Hello, World!"}
    }
  ]
}</code></pre>
    </div>
    <div class="response-section">
      <h4>Expected Response</h4>
      <pre><code>{
  "id": "cmd-456",
  "name": "hello-world",
  "status": "created",
  "created_at": "2025-01-01T12:00:00Z",
  "execution_url": "/api/v1/commands/cmd-456/execute"
}</code></pre>
    </div>
    <button class="try-button">Try It Live</button>
  </div>
</div>

## 🔧 SDK & Libraries

Official SDKs and community libraries for popular programming languages:

<div class="sdk-grid">
  <div class="sdk-card">
    <div class="sdk-icon">🐍</div>
    <h4>Python SDK</h4>
    <p>Official Python client with full feature support</p>
    <code>pip install leanvibe-sdk</code>
    <a href="/api/sdks/python" class="sdk-link">Documentation →</a>
  </div>
  
  <div class="sdk-card">
    <div class="sdk-icon">📟</div>
    <h4>JavaScript SDK</h4>
    <p>Browser and Node.js support with TypeScript definitions</p>
    <code>npm install @leanvibe/sdk</code>
    <a href="/api/sdks/javascript" class="sdk-link">Documentation →</a>
  </div>
  
  <div class="sdk-card">
    <div class="sdk-icon">🦀</div>
    <h4>Rust SDK</h4>
    <p>High-performance async client for Rust applications</p>
    <code>cargo add leanvibe</code>
    <a href="/api/sdks/rust" class="sdk-link">Documentation →</a>
  </div>
  
  <div class="sdk-card">
    <div class="sdk-icon">☕</div>
    <h4>Go SDK</h4>
    <p>Idiomatic Go client with comprehensive examples</p>
    <code>go get github.com/leanvibe/go-sdk</code>
    <a href="/api/sdks/go" class="sdk-link">Documentation →</a>
  </div>
</div>

## 📖 Code Examples

### Python Example

```python
from leanvibe import LeanVibeClient

# Initialize client
client = LeanVibeClient(api_key="your_api_key")

# Create a custom command
command = client.commands.create({
    "name": "feature-development",
    "description": "Full feature development workflow",
    "agents": ["architect", "developer", "tester"],
    "steps": [
        {
            "name": "design",
            "agent": "architect", 
            "action": "create_design",
            "params": {"feature": "user_authentication"}
        },
        {
            "name": "implement",
            "agent": "developer",
            "action": "write_code",
            "depends_on": ["design"]
        },
        {
            "name": "test",
            "agent": "tester",
            "action": "create_tests",
            "depends_on": ["implement"]
        }
    ]
})

# Execute the command
execution = client.commands.execute(command.id)

# Monitor progress
for update in client.commands.stream_updates(execution.id):
    print(f"Status: {update.status}, Progress: {update.progress}%")
```

### JavaScript Example

```javascript
import { LeanVibeClient } from '@leanvibe/sdk';

// Initialize client
const client = new LeanVibeClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.leanvibe.dev'
});

// Create and execute command
async function runAutonomousWorkflow() {
  try {
    // Create command
    const command = await client.commands.create({
      name: 'bug-fix-workflow',
      description: 'Automated bug fixing process',
      agents: ['developer', 'tester'],
      steps: [
        {
          name: 'analyze',
          agent: 'developer',
          action: 'analyze_bug',
          params: { issue_id: 'BUG-123' }
        },
        {
          name: 'fix',
          agent: 'developer', 
          action: 'implement_fix',
          depends_on: ['analyze']
        },
        {
          name: 'verify',
          agent: 'tester',
          action: 'verify_fix',
          depends_on: ['fix']
        }
      ]
    });

    // Execute command
    const execution = await client.commands.execute(command.id);
    
    // Stream real-time updates
    const stream = client.commands.streamUpdates(execution.id);
    
    for await (const update of stream) {
      console.log(`${update.step}: ${update.status}`);
      
      if (update.status === 'completed') {
        console.log('Bug fix completed successfully!');
        break;
      }
    }
    
  } catch (error) {
    console.error('Workflow failed:', error);
  }
}

runAutonomousWorkflow();
```

## 🚨 Error Handling

The API uses conventional HTTP response codes and provides detailed error messages:

### HTTP Status Codes

- `200` - Success
- `201` - Created successfully
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid API key)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_COMMAND",
    "message": "Command validation failed",
    "details": {
      "field": "agents",
      "reason": "At least one agent is required"
    },
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `INVALID_API_KEY` | API key is missing or invalid | Check your API key and permissions |
| `RATE_LIMITED` | Too many requests | Implement exponential backoff |
| `AGENT_UNAVAILABLE` | Requested agent is not available | Check agent status or try different agent |
| `COMMAND_FAILED` | Command execution failed | Check command syntax and parameters |
| `INSUFFICIENT_PERMISSIONS` | Missing required permissions | Contact your administrator |

## 📈 Rate Limits

API rate limits ensure fair usage and system stability:

### Default Limits

- **Free Tier**: 100 requests/hour
- **Professional**: 1,000 requests/hour  
- **Enterprise**: 10,000 requests/hour
- **Custom**: Negotiated limits

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

### Best Practices

- Implement exponential backoff for retries
- Cache responses when appropriate
- Use webhooks for real-time updates instead of polling
- Monitor your usage through the dashboard

## 📚 Next Steps

Now that you understand the API basics, explore specific endpoints:

<div class="next-steps">
  <a href="/api/commands/" class="next-step-card">
    <h4>🎯 Custom Commands</h4>
    <p>Create sophisticated multi-agent workflows</p>
  </a>
  
  <a href="/api/coordination/" class="next-step-card">
    <h4>🤖 Multi-Agent Coordination</h4>
    <p>Orchestrate agent collaboration</p>
  </a>
  
  <a href="/api/observability/" class="next-step-card">
    <h4>📊 Observability</h4>
    <p>Monitor and debug your systems</p>
  </a>
  
  <a href="/examples/integrations/" class="next-step-card">
    <h4>🔧 Integration Examples</h4>
    <p>Real-world implementation patterns</p>
  </a>
</div>

<style>
.quick-start-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.start-step {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
}

.start-step h3 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.start-step p {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-2);
}

.start-step code {
  display: block;
  padding: 0.5rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 4px;
  font-size: 0.875rem;
}

.api-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.stat-card {
  text-align: center;
  padding: 1.5rem 1rem;
  background: var(--lv-glass-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--lv-glass-border);
  border-radius: 8px;
}

.stat-number {
  display: block;
  font-size: 2rem;
  font-weight: 700;
  color: var(--lv-primary);
  margin-bottom: 0.5rem;
}

.stat-label {
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
}

.api-explorer {
  margin: 2rem 0;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
}

.api-explorer h3 {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
}

.explorer-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--vp-c-border);
}

.tab-button {
  padding: 0.5rem 1rem;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s ease;
}

.tab-button.active {
  color: var(--lv-primary);
  border-bottom-color: var(--lv-primary);
}

.tab-content {
  display: none;
}

.tab-content.active {
  display: block;
}

.request-section, .response-section {
  margin-bottom: 1rem;
}

.request-section h4, .response-section h4 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
  font-size: 1rem;
}

.request-section pre, .response-section pre {
  margin: 0;
  padding: 1rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 6px;
  overflow-x: auto;
}

.try-button {
  padding: 0.5rem 1rem;
  background: var(--lv-gradient-primary);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
}

.try-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.sdk-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.sdk-card {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  text-align: center;
}

.sdk-icon {
  font-size: 2rem;
  margin-bottom: 1rem;
}

.sdk-card h4 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.sdk-card p {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
}

.sdk-card code {
  display: block;
  margin-bottom: 1rem;
  padding: 0.5rem;
  background: var(--vp-c-bg);
  border-radius: 4px;
  font-size: 0.8rem;
}

.sdk-link {
  color: var(--lv-primary);
  text-decoration: none;
  font-weight: 500;
}

.sdk-link:hover {
  text-decoration: underline;
}

.next-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.next-step-card {
  display: block;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  text-decoration: none;
  color: var(--vp-c-text-1);
  transition: all 0.2s ease;
}

.next-step-card:hover {
  border-color: var(--lv-primary);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

.next-step-card h4 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.next-step-card p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
}

@media (max-width: 768px) {
  .quick-start-section {
    grid-template-columns: 1fr;
  }
  
  .api-stats {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .sdk-grid {
    grid-template-columns: 1fr;
  }
  
  .explorer-tabs {
    flex-direction: column;
  }
  
  .tab-button {
    text-align: left;
  }
}
</style>

<script setup>
// Tab functionality for API explorer
import { onMounted } from 'vue'

onMounted(() => {
  const tabButtons = document.querySelectorAll('.tab-button')
  const tabContents = document.querySelectorAll('.tab-content')
  
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const tabName = button.dataset.tab
      
      // Remove active class from all tabs
      tabButtons.forEach(btn => btn.classList.remove('active'))
      tabContents.forEach(content => content.classList.remove('active'))
      
      // Add active class to clicked tab
      button.classList.add('active')
      document.getElementById(`${tabName}-tab`).classList.add('active')
    })
  })
  
  // Try buttons functionality
  const tryButtons = document.querySelectorAll('.try-button')
  tryButtons.forEach(button => {
    button.addEventListener('click', () => {
      button.textContent = 'Executing...'
      setTimeout(() => {
        button.textContent = 'Success! ✅'
        setTimeout(() => {
          button.textContent = 'Try It Live'
        }, 2000)
      }, 1000)
    })
  })
})
</script>
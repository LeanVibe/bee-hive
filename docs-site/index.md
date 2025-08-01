---
layout: home
title: LeanVibe Agent Hive
titleTemplate: Autonomous Software Development Platform

hero:
  name: LeanVibe Agent Hive
  text: Autonomous Software Development That Actually Works
  tagline: Watch AI agents build complete features from start to finish. Transform your development workflow with multi-agent coordination, intelligent automation, and production-ready results.
  image:
    src: /images/hero-agent-coordination.svg
    alt: Multi-agent coordination visualization
  actions:
    - theme: brand
      text: Try Live Demo
      link: /demo
    - theme: alt
      text: Quick Setup
      link: /learn/getting-started/setup
    - theme: alt
      text: Watch Video
      link: /examples/showcase/featured

features:
  - icon: 🤖
    title: Multi-Agent Coordination
    details: Architect, developer, tester, and reviewer agents collaborate seamlessly on complex projects with intelligent task distribution and conflict resolution.
    link: /learn/fundamentals/coordination
    
  - icon: ⚡
    title: Custom Commands System
    details: Create sophisticated multi-step workflows with declarative YAML definitions. 8 advanced commands for every development scenario.
    link: /api/commands/
    
  - icon: 🎯
    title: Production-Ready Platform
    details: Enterprise-grade security, real-time monitoring, GitHub integration, and 9.5/10 quality score validated by external AI assessment.
    link: /enterprise/
    
  - icon: 🚀
    title: <2 Minute Setup
    details: DevContainer, professional, or sandbox modes. Zero-configuration autonomous development ready in minutes, not hours.
    link: /learn/getting-started/setup
    
  - icon:  🧠
    title: Context Memory
    details: Agents learn and remember your project patterns, coding styles, and preferences for increasingly intelligent automation.
    link: /learn/fundamentals/memory
    
  - icon: 📊
    title: Real-Time Observability
    details: Live dashboards, WebSocket streams, comprehensive metrics, and intelligent alerting for complete system visibility.
    link: /api/observability/
---

<!-- Live Metrics Widget -->
<MetricsWidget />

<!-- Interactive Feature Showcase -->
<FeatureShowcase />

<!-- Code Playground Demo -->
<div class="vp-doc" style="margin-top: 3rem;">

## See It Working Right Now

Experience autonomous development in action with our interactive playground. No setup required.

<CodePlayground />

</div>

<!-- Live Agent Coordination Demo -->
<div class="vp-doc" style="margin-top: 2rem;">

## Multi-Agent Coordination in Action

Watch specialized AI agents collaborate on real development tasks in real-time.

<AgentDemo />

</div>

<!-- Success Stories Section -->
<div class="vp-doc" style="margin-top: 3rem;">

## Trusted by Forward-Thinking Teams

<div class="success-stats">
  <div class="stat-card">
    <div class="stat-number">9.5/10</div>
    <div class="stat-label">Quality Score</div>
    <div class="stat-desc">External AI validation</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">100%</div>
    <div class="stat-label">Success Rate</div>
    <div class="stat-desc">In testing environments</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">65%</div>
    <div class="stat-label">Faster Setup</div>
    <div class="stat-desc">Compared to alternatives</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">5,365</div>
    <div class="stat-label">Lines of Code</div>
    <div class="stat-desc">Advanced custom commands</div>
  </div>
</div>

### What Makes This Different

- **Actually Working**: Not just demos—production-ready autonomous development
- **Multi-Agent Intelligence**: Specialized agents that collaborate like a real team
- **Enterprise Grade**: Security, compliance, and scalability built-in
- **Community Driven**: Open source with active community and continuous innovation

</div>

<!-- Getting Started Section -->
<div class="vp-doc" style="margin-top: 3rem;">

## Choose Your Experience

<div class="setup-options">
  <div class="setup-card setup-recommended">
    <div class="setup-badge">Recommended</div>
    <h3>🎯 DevContainer</h3>
    <div class="setup-time">&lt;2 minutes</div>
    <p>Zero-configuration autonomous development. Just open in VS Code.</p>
    <div class="setup-steps">
      <code>git clone https://github.com/LeanVibe/bee-hive.git<br>
      code bee-hive  # Click "Reopen in Container"</code>
    </div>
    <a href="/learn/getting-started/devcontainer" class="setup-button">Get Started</a>
  </div>
  
  <div class="setup-card">
    <h3>⚡ Professional</h3>
    <div class="setup-time">&lt;5 minutes</div>
    <p>Enterprise-grade one-command setup with full control.</p>
    <div class="setup-steps">
      <code>git clone https://github.com/LeanVibe/bee-hive.git<br>
      make setup && make start</code>
    </div>
    <a href="/learn/getting-started/professional" class="setup-button">Learn More</a>
  </div>
  
  <div class="setup-card">
    <h3>🎮 Sandbox</h3>
    <div class="setup-time">0 minutes</div>
    <p>Try autonomous development in your browser. No installation.</p>
    <div class="setup-steps">
      <em>Interactive browser-based environment</em>
    </div>
    <a href="/demo" target="_blank" class="setup-button">Try Now</a>
  </div>
</div>

</div>

<!-- Community Hub -->
<CommunityHub />

<style>
.success-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.stat-card {
  text-align: center;
  padding: 2rem 1rem;
  background: var(--lv-glass-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--lv-glass-border);
  border-radius: 12px;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--lv-shadow-xl);
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 700;
  background: var(--lv-gradient-primary);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
}

.stat-label {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.25rem;
}

.stat-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.setup-options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.setup-card {
  position: relative;
  padding: 2rem;
  background: var(--vp-c-bg-soft);
  border: 2px solid var(--vp-c-border);
  border-radius: 12px;
  transition: all 0.3s ease;
}

.setup-card:hover {
  border-color: var(--lv-primary);
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
}

.setup-recommended {
  border-color: var(--lv-primary);
  background: rgba(99, 102, 241, 0.05);
}

.setup-badge {
  position: absolute;
  top: -0.75rem;
  left: 1.5rem;
  padding: 0.25rem 0.75rem;
  background: var(--lv-gradient-primary);
  color: white;
  font-size: 0.75rem;
  font-weight: 600;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.setup-card h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.25rem;
  color: var(--vp-c-text-1);
}

.setup-time {
  font-size: 0.875rem;
  color: var(--lv-secondary);
  font-weight: 600;
  margin-bottom: 1rem;
}

.setup-card p {
  color: var(--vp-c-text-2);
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.setup-steps {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1.5rem;
  font-family: var(--vp-font-family-mono);
  font-size: 0.875rem;
}

.setup-steps code {
  background: none;
  padding: 0;
  border-radius: 0;
  font-size: inherit;
  color: var(--vp-c-text-1);
  line-height: 1.6;
}

.setup-steps em {
  color: var(--vp-c-text-2);
  text-align: center;
  display: block;
}

.setup-button {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  background: var(--lv-gradient-primary);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.2s ease;
}

.setup-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
}

@media (max-width: 768px) {
  .success-stats {
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }
  
  .stat-card {
    padding: 1.5rem 1rem;
  }
  
  .stat-number {
    font-size: 2rem;
  }
  
  .setup-options {
    grid-template-columns: 1fr;
  }
  
  .setup-card {
    padding: 1.5rem;
  }
}
</style>
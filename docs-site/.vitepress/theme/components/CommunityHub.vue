<template>
  <div class="community-hub">
    <div class="hub-header">
      <h2>Join the Community</h2>
      <p>Connect with developers building the future of autonomous software development</p>
    </div>
    
    <div class="community-stats">
      <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-number">{{ stats.members.toLocaleString() }}</div>
        <div class="stat-label">Community Members</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🚀</div>
        <div class="stat-number">{{ stats.projects }}</div>
        <div class="stat-label">Active Projects</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">💡</div>
        <div class="stat-number">{{ stats.contributions }}</div>
        <div class="stat-label">Contributions</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">⭐</div>
        <div class="stat-number">{{ stats.stars.toLocaleString() }}</div>
        <div class="stat-label">GitHub Stars</div>
      </div>
    </div>
    
    <div class="community-sections">
      <div class="section-card">
        <div class="section-icon">💬</div>
        <h3>Discord Community</h3>
        <p>Join our active Discord server for real-time discussions, support, and collaboration.</p>
        <div class="section-features">
          <span class="feature-tag">24/7 Support</span>
          <span class="feature-tag">Office Hours</span>
          <span class="feature-tag">Live Demos</span>
        </div>
        <a href="https://discord.gg/leanvibe" target="_blank" class="section-button">
          Join Discord
        </a>
      </div>
      
      <div class="section-card">
        <div class="section-icon">🎯</div>
        <h3>GitHub Discussions</h3>
        <p>Share ideas, ask questions, and collaborate on the project's future direction.</p>
        <div class="section-features">
          <span class="feature-tag">Q&A Forum</span>
          <span class="feature-tag">Feature Requests</span>
          <span class="feature-tag">Show & Tell</span>
        </div>
        <a href="https://github.com/LeanVibe/bee-hive/discussions" target="_blank" class="section-button">
          Join Discussions
        </a>
      </div>
      
      <div class="section-card">
        <div class="section-icon">🛠️</div>
        <h3>Contribute</h3>
        <p>Help build the future of autonomous development. Contribute code, docs, or ideas.</p>
        <div class="section-features">
          <span class="feature-tag">Open Source</span>
          <span class="feature-tag">All Levels</span>
          <span class="feature-tag">Recognition</span>
        </div>
        <a href="/community/contributing" class="section-button">
          Start Contributing
        </a>
      </div>
    </div>
    
    <div class="recent-activity">
      <h3>Recent Community Activity</h3>
      <div class="activity-feed">
        <div 
          v-for="activity in recentActivity" 
          :key="activity.id"
          class="activity-item"
        >
          <div class="activity-avatar">
            <img :src="activity.avatar" :alt="activity.author" />
          </div>
          <div class="activity-content">
            <div class="activity-header">
              <span class="activity-author">{{ activity.author }}</span>
              <span class="activity-action">{{ activity.action }}</span>
              <span class="activity-time">{{ formatTime(activity.timestamp) }}</span>
            </div>
            <div class="activity-title">{{ activity.title }}</div>
            <div v-if="activity.description" class="activity-description">
              {{ activity.description }}
            </div>
          </div>
          <div class="activity-type">
            <span :class="['type-badge', activity.type]">{{ activity.type }}</span>
          </div>
        </div>
      </div>
      
      <div class="activity-footer">
        <a href="/community/activity" class="view-all-link">
          View All Activity →
        </a>
      </div>
    </div>
    
    <div class="newsletter-signup">
      <div class="newsletter-content">
        <h3>Stay Updated</h3>
        <p>Get the latest updates on autonomous development, new features, and community highlights.</p>
        
        <form @submit.prevent="subscribeNewsletter" class="newsletter-form">
          <input 
            v-model="email"
            type="email" 
            placeholder="Enter your email"
            class="email-input"
            required
          />
          <button 
            type="submit" 
            :disabled="isSubscribing"
            class="subscribe-button"
          >
            {{ isSubscribing ? 'Subscribing...' : 'Subscribe' }}
          </button>
        </form>
        
        <div class="newsletter-features">
          <span class="newsletter-feature">📧 Weekly Updates</span>
          <span class="newsletter-feature">🎯 No Spam</span>
          <span class="newsletter-feature">🔓 Unsubscribe Anytime</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

// Component state
const email = ref('')
const isSubscribing = ref(false)

// Community stats
const stats = ref({
  members: 12547,
  projects: 234,
  contributions: 1892,
  stars: 8934
})

// Recent activity data
const recentActivity = ref([
  {
    id: 1,
    author: 'Sarah Chen',
    avatar: '/images/avatars/sarah.jpg',
    action: 'opened a discussion',
    title: 'Best practices for multi-agent workflows',
    description: 'Looking for advice on optimizing agent coordination for large projects...',
    type: 'discussion',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000) // 2 hours ago
  },
  {
    id: 2,
    author: 'Mike Rodriguez',
    avatar: '/images/avatars/mike.jpg',
    action: 'merged a pull request',
    title: 'Add performance monitoring to custom commands',
    description: null,
    type: 'pr',
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000) // 4 hours ago
  },
  {
    id: 3,
    author: 'Anna Thompson',
    avatar: '/images/avatars/anna.jpg',
    action: 'shared a project',
    title: 'E-commerce automation with LeanVibe agents',
    description: 'Built a full e-commerce platform using autonomous agents for development...',
    type: 'showcase',
    timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000) // 6 hours ago
  },
  {
    id: 4,
    author: 'David Park',
    avatar: '/images/avatars/david.jpg',
    action: 'opened an issue',
    title: 'Feature request: Custom agent types',
    description: 'Would love to see support for custom agent types in the coordination system...',
    type: 'issue',
    timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000) // 8 hours ago
  }
])

// Methods
const formatTime = (timestamp: Date) => {
  const now = new Date()
  const diff = now.getTime() - timestamp.getTime()
  const hours = Math.floor(diff / (1000 * 60 * 60))
  
  if (hours < 1) {
    const minutes = Math.floor(diff / (1000 * 60))
    return `${minutes}m ago`
  } else if (hours < 24) {
    return `${hours}h ago`
  } else {
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }
}

const subscribeNewsletter = async () => {
  if (!email.value) return
  
  isSubscribing.value = true
  
  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Show success message (in a real app, you'd integrate with your newsletter service)
    alert('Thanks for subscribing! Check your email for confirmation.')
    email.value = ''
  } catch (error) {
    alert('Subscription failed. Please try again.')
  } finally {
    isSubscribing.value = false
  }
}

// Simulate real-time stats updates
onMounted(() => {
  setInterval(() => {
    // Small random fluctuations in stats
    stats.value.members += Math.floor(Math.random() * 3)
    stats.value.contributions += Math.floor(Math.random() * 2)
    stats.value.stars += Math.floor(Math.random() * 2)
  }, 30000) // Update every 30 seconds
})
</script>

<style scoped>
.community-hub {
  margin: 3rem 0;
  padding: 2rem;
  background: var(--lv-glass-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--lv-glass-border);
  border-radius: 16px;
  box-shadow: var(--lv-shadow-xl);
}

.hub-header {
  text-align: center;
  margin-bottom: 2rem;
}

.hub-header h2 {
  margin: 0 0 0.5rem 0;
  font-size: 2rem;
  font-weight: 700;
  background: var(--lv-gradient-primary);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hub-header p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 1.1rem;
}

.community-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minMax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  text-align: center;
  padding: 1.5rem 1rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--lv-shadow-lg);
}

.stat-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.stat-number {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--lv-primary);
  margin-bottom: 0.25rem;
}

.stat-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.community-sections {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.section-card {
  padding: 2rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  text-align: center;
  transition: all 0.3s ease;
}

.section-card:hover {
  border-color: var(--lv-primary);
  box-shadow: var(--lv-shadow-lg);
}

.section-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.section-card h3 {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
}

.section-card p {
  margin: 0 0 1.5rem 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.section-features {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}

.feature-tag {
  padding: 0.25rem 0.5rem;
  background: var(--lv-gradient-primary);
  color: white;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
}

.section-button {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  background: var(--lv-gradient-primary);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.2s ease;
}

.section-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
}

.recent-activity {
  margin-bottom: 2rem;
}

.recent-activity h3 {
  margin: 0 0 1.5rem 0;
  color: var(--vp-c-text-1);
  text-align: center;
}

.activity-feed {
  space-y: 1rem;
}

.activity-item {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  margin-bottom: 1rem;
  transition: all 0.2s ease;
}

.activity-item:hover {
  border-color: var(--lv-primary);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

.activity-avatar img {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--lv-gradient-primary);
}

.activity-content {
  flex: 1;
  min-width: 0;
}

.activity-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
  flex-wrap: wrap;
}

.activity-author {
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.activity-action {
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
}

.activity-time {
  color: var(--vp-c-text-3);
  font-size: 0.75rem;
  margin-left: auto;
}

.activity-title {
  font-weight: 500;
  color: var(--vp-c-text-1);
  margin-bottom: 0.25rem;
}

.activity-description {
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
  line-height: 1.4;
}

.activity-type {
  display: flex;
  align-items: flex-start;
}

.type-badge {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
}

.type-badge.discussion {
  background: rgba(99, 102, 241, 0.1);
  color: var(--lv-primary);
}

.type-badge.pr {
  background: rgba(16, 185, 129, 0.1);
  color: var(--lv-secondary);
}

.type-badge.showcase {
  background: rgba(245, 158, 11, 0.1);
  color: var(--lv-accent);
}

.type-badge.issue {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
}

.activity-footer {
  text-align: center;
  margin-top: 1.5rem;
}

.view-all-link {
  color: var(--lv-primary);
  text-decoration: none;
  font-weight: 500;
}

.view-all-link:hover {
  text-decoration: underline;
}

.newsletter-signup {
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
}

.newsletter-content h3 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.newsletter-content p {
  margin: 0 0 1.5rem 0;
  color: var(--vp-c-text-2);
}

.newsletter-form {
  display: flex;
  gap: 1rem;
  max-width: 400px;
  margin: 0 auto 1rem auto;
}

.email-input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.875rem;
}

.email-input:focus {
  outline: none;
  border-color: var(--lv-primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.subscribe-button {
  padding: 0.75rem 1.5rem;
  background: var(--lv-gradient-primary);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.subscribe-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.subscribe-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.newsletter-features {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.newsletter-feature {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

@media (max-width: 768px) {
  .community-stats {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .community-sections {
    grid-template-columns: 1fr;
  }
  
  .activity-item {
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .activity-header {
    justify-content: space-between;
  }
  
  .activity-time {
    margin-left: 0;
  }
  
  .newsletter-form {
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .newsletter-features {
    flex-direction: column;
    gap: 0.5rem;
  }
}
</style>
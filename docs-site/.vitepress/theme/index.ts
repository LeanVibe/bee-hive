import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

// Custom components
import CodePlayground from './components/CodePlayground.vue'
import AgentDemo from './components/AgentDemo.vue'
import FeatureShowcase from './components/FeatureShowcase.vue'
import MetricsWidget from './components/MetricsWidget.vue'
import CommunityHub from './components/CommunityHub.vue'
import EnterpriseCard from './components/EnterpriseCard.vue'
import InteractiveGuide from './components/InteractiveGuide.vue'
import LiveDemo from './components/LiveDemo.vue'

// Custom layouts
import HomeLayout from './layouts/HomeLayout.vue'
import EnterpriseLayout from './layouts/EnterpriseLayout.vue'

// Custom styles
import './styles/main.css'
import './styles/components.css'
import './styles/syntax.css'

export default {
  extends: DefaultTheme,
  
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // Custom layout slots
      'home-hero-before': () => h(MetricsWidget),
      'home-features-after': () => h(FeatureShowcase),
      'doc-footer-before': () => h(CommunityHub),
      'nav-bar-content-after': () => h(LiveDemo),
    })
  },
  
  enhanceApp({ app, router, siteData }) {
    // Register global components
    app.component('CodePlayground', CodePlayground)
    app.component('AgentDemo', AgentDemo)
    app.component('FeatureShowcase', FeatureShowcase)
    app.component('MetricsWidget', MetricsWidget)
    app.component('CommunityHub', CommunityHub)
    app.component('EnterpriseCard', EnterpriseCard)
    app.component('InteractiveGuide', InteractiveGuide)
    app.component('LiveDemo', LiveDemo)
    
    // Register custom layouts
    app.component('HomeLayout', HomeLayout)
    app.component('EnterpriseLayout', EnterpriseLayout)
    
    // Global app configuration
    app.config.globalProperties.$leanvibe = {
      version: siteData.value.themeConfig?.version || '2.0.0',
      apiUrl: 'https://api.leanvibe.dev',
      demoUrl: 'https://demo.leanvibe.dev'
    }
    
    // Router configuration
    router.beforeEach((to, from, next) => {
      // Track page views
      if (typeof window !== 'undefined' && window.fathom) {
        window.fathom('trackPageview')
      }
      next()
    })
  }
} satisfies Theme
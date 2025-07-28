/**
 * Session Color Manager
 * 
 * Provides consistent color coding for development sessions and agents
 * with visual identity management and performance heatmap colors.
 */

import { scaleOrdinal, scaleSequential } from 'd3-scale'
import { interpolateViridis, interpolateRdYlBu, interpolatePlasma } from 'd3-scale-chromatic'
import { SessionInfo, AgentInfo, SecurityRisk } from '@/types/hooks'

export interface ColorScheme {
  primary: string
  secondary: string
  accent: string
  background: string
  border: string
  text: string
  textSecondary: string
}

export interface PerformanceColorConfig {
  excellent: string // 90-100%
  good: string      // 70-89%
  average: string   // 50-69%
  poor: string      // 30-49%
  critical: string  // 0-29%
}

export interface SessionColorConfig {
  active: string
  completed: string
  error: string
  terminated: string
  idle: string
}

export class SessionColorManager {
  private sessionColors = new Map<string, ColorScheme>()
  private agentColors = new Map<string, string>()
  private colorPalettes: string[][] = []
  private currentPaletteIndex = 0
  
  // Predefined color palettes for sessions
  private readonly sessionPalettes = [
    // Vibrant tech colors
    ['#3B82F6', '#1E40AF', '#DBEAFE', '#F8FAFC', '#E5E7EB', '#1F2937', '#6B7280'],
    ['#10B981', '#047857', '#D1FAE5', '#F0FDF4', '#E5E7EB', '#1F2937', '#6B7280'],
    ['#F59E0B', '#D97706', '#FEF3C7', '#FFFBEB', '#E5E7EB', '#1F2937', '#6B7280'],
    ['#EF4444', '#DC2626', '#FEE2E2', '#FEF2F2', '#E5E7EB', '#1F2937', '#6B7280'],
    ['#8B5CF6', '#7C3AED', '#EDE9FE', '#FAF7FF', '#E5E7EB', '#1F2937', '#6B7280'],
    ['#EC4899', '#DB2777', '#FCE7F3', '#FDF2F8', '#E5E7EB', '#1F2937', '#6B7280'],
    ['#06B6D4', '#0891B2', '#CFFAFE', '#F0FDFF', '#E5E7EB', '#1F2937', '#6B7280'],
    ['#84CC16', '#65A30D', '#ECFCCB', '#F7FEE7', '#E5E7EB', '#1F2937', '#6B7280'],
  ]
  
  // Performance color configurations
  private readonly performanceColors: PerformanceColorConfig = {
    excellent: '#10B981', // Green
    good: '#3B82F6',      // Blue
    average: '#F59E0B',   // Amber
    poor: '#EF4444',      // Red
    critical: '#7F1D1D'   // Dark red
  }
  
  // Session status colors
  private readonly sessionStatusColors: SessionColorConfig = {
    active: '#10B981',    // Green
    completed: '#6B7280', // Gray
    error: '#EF4444',     // Red
    terminated: '#7C2D12', // Dark red
    idle: '#F59E0B'       // Amber
  }
  
  // Security risk colors
  private readonly securityRiskColors = {
    [SecurityRisk.SAFE]: '#10B981',
    [SecurityRisk.LOW]: '#84CC16',
    [SecurityRisk.MEDIUM]: '#F59E0B',
    [SecurityRisk.HIGH]: '#EF4444',
    [SecurityRisk.CRITICAL]: '#7F1D1D'
  }
  
  constructor() {
    this.initializePalettes()
  }
  
  private initializePalettes(): void {
    this.colorPalettes = this.sessionPalettes.map(palette => [...palette])
  }
  
  /**
   * Get or create a color scheme for a session
   */
  getSessionColorScheme(sessionId: string): ColorScheme {
    if (this.sessionColors.has(sessionId)) {
      return this.sessionColors.get(sessionId)!
    }
    
    const palette = this.getNextPalette()
    const colorScheme: ColorScheme = {
      primary: palette[0],
      secondary: palette[1],
      accent: palette[0],
      background: palette[2],
      border: palette[4],
      text: palette[5],
      textSecondary: palette[6]
    }
    
    this.sessionColors.set(sessionId, colorScheme)
    return colorScheme
  }
  
  /**
   * Get session color by status
   */
  getSessionStatusColor(status: SessionInfo['status']): string {
    return this.sessionStatusColors[status] || this.sessionStatusColors.active
  }
  
  /**
   * Get or create a color for an agent
   */
  getAgentColor(agentId: string, sessionId?: string): string {
    if (this.agentColors.has(agentId)) {
      return this.agentColors.get(agentId)!
    }
    
    let color: string
    if (sessionId) {
      const sessionScheme = this.getSessionColorScheme(sessionId)
      color = sessionScheme.primary
    } else {
      const palette = this.getNextPalette()
      color = palette[0]
    }
    
    this.agentColors.set(agentId, color)
    return color
  }
  
  /**
   * Get performance-based color
   */
  getPerformanceColor(performanceScore: number): string {
    if (performanceScore >= 90) return this.performanceColors.excellent
    if (performanceScore >= 70) return this.performanceColors.good
    if (performanceScore >= 50) return this.performanceColors.average
    if (performanceScore >= 30) return this.performanceColors.poor
    return this.performanceColors.critical
  }
  
  /**
   * Get security risk color
   */
  getSecurityRiskColor(risk: SecurityRisk): string {
    return this.securityRiskColors[risk]
  }
  
  /**
   * Generate a continuous color scale for performance heatmaps
   */
  getPerformanceHeatmapScale(domain: [number, number] = [0, 100]) {
    return scaleSequential(interpolateViridis).domain(domain)
  }
  
  /**
   * Generate a color scale for agent activity heatmap
   */
  getActivityHeatmapScale(domain: [number, number]) {
    return scaleSequential(interpolateRdYlBu).domain(domain)
  }
  
  /**
   * Get error intensity color scale
   */
  getErrorIntensityScale(domain: [number, number]) {
    return scaleSequential(interpolatePlasma).domain(domain)
  }
  
  /**
   * Get the next color palette in rotation
   */
  private getNextPalette(): string[] {
    const palette = this.colorPalettes[this.currentPaletteIndex]
    this.currentPaletteIndex = (this.currentPaletteIndex + 1) % this.colorPalettes.length
    return palette
  }
  
  /**
   * Generate gradient CSS for backgrounds
   */
  generateGradient(sessionId: string, direction: 'horizontal' | 'vertical' | 'radial' = 'horizontal'): string {
    const scheme = this.getSessionColorScheme(sessionId)
    
    switch (direction) {
      case 'horizontal':
        return `linear-gradient(90deg, ${scheme.primary}20, ${scheme.secondary}20)`
      case 'vertical':
        return `linear-gradient(180deg, ${scheme.primary}20, ${scheme.secondary}20)`
      case 'radial':
        return `radial-gradient(circle, ${scheme.primary}20, ${scheme.secondary}20)`
      default:
        return `linear-gradient(90deg, ${scheme.primary}20, ${scheme.secondary}20)`
    }
  }
  
  /**
   * Get contrasting text color for a background
   */
  getContrastingTextColor(backgroundColor: string): string {
    // Simple contrast calculation - can be enhanced with proper color theory
    const hex = backgroundColor.replace('#', '')
    const r = parseInt(hex.substr(0, 2), 16)
    const g = parseInt(hex.substr(2, 2), 16)
    const b = parseInt(hex.substr(4, 2), 16)
    
    // Calculate luminance
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    return luminance > 0.5 ? '#1F2937' : '#F9FAFB'
  }
  
  /**
   * Create color scale for categorical data
   */
  createCategoricalScale(categories: string[]): (category: string) => string {
    const colors = this.sessionPalettes.flat().filter(color => color.startsWith('#'))
    return scaleOrdinal<string>()
      .domain(categories)
      .range(colors)
  }
  
  /**
   * Get all session colors for legend
   */
  getAllSessionColors(): Array<{ sessionId: string; color: ColorScheme }> {
    return Array.from(this.sessionColors.entries()).map(([sessionId, color]) => ({
      sessionId,
      color
    }))
  }
  
  /**
   * Reset color assignments (useful for testing or cleanup)
   */
  reset(): void {
    this.sessionColors.clear()
    this.agentColors.clear()
    this.currentPaletteIndex = 0
  }
  
  /**
   * Get color palette for a specific theme
   */
  getThemeColors(theme: 'light' | 'dark' = 'light') {
    if (theme === 'dark') {
      return {
        background: '#1F2937',
        surface: '#374151',
        border: '#4B5563',
        text: '#F9FAFB',
        textSecondary: '#D1D5DB',
        primary: '#3B82F6',
        success: '#10B981',
        warning: '#F59E0B',
        error: '#EF4444'
      }
    }
    
    return {
      background: '#F9FAFB',
      surface: '#FFFFFF',
      border: '#E5E7EB',
      text: '#1F2937',
      textSecondary: '#6B7280',
      primary: '#3B82F6',
      success: '#10B981',
      warning: '#F59E0B',
      error: '#EF4444'
    }
  }
}

// Singleton instance
export const sessionColorManager = new SessionColorManager()

// Utility functions for Vue components
export const useSessionColors = () => {
  return {
    getSessionColor: (sessionId: string) => sessionColorManager.getSessionColorScheme(sessionId),
    getAgentColor: (agentId: string, sessionId?: string) => sessionColorManager.getAgentColor(agentId, sessionId),
    getPerformanceColor: (score: number) => sessionColorManager.getPerformanceColor(score),
    getSecurityRiskColor: (risk: SecurityRisk) => sessionColorManager.getSecurityRiskColor(risk),
    getSessionStatusColor: (status: SessionInfo['status']) => sessionColorManager.getSessionStatusColor(status),
    generateGradient: (sessionId: string, direction?: 'horizontal' | 'vertical' | 'radial') => 
      sessionColorManager.generateGradient(sessionId, direction),
    createHeatmapScale: (domain: [number, number]) => sessionColorManager.getPerformanceHeatmapScale(domain),
    getThemeColors: (theme: 'light' | 'dark') => sessionColorManager.getThemeColors(theme)
  }
}
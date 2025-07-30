/**
 * Responsive Breakpoints Composable
 * 
 * Provides reactive breakpoint detection for responsive design
 * based on Tailwind CSS breakpoints.
 */

import { ref, computed, onMounted, onUnmounted } from 'vue'

export interface BreakpointConfig {
  sm: number
  md: number
  lg: number
  xl: number
  '2xl': number
}

export type ScreenSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl'

const defaultBreakpoints: BreakpointConfig = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536
}

export function useBreakpoints(breakpoints: BreakpointConfig = defaultBreakpoints) {
  const windowWidth = ref(0)

  const updateWindowWidth = () => {
    windowWidth.value = window.innerWidth
  }

  // Screen size detection
  const screenSize = computed<ScreenSize>(() => {
    const width = windowWidth.value
    
    if (width >= breakpoints['2xl']) return '2xl'
    if (width >= breakpoints.xl) return 'xl'
    if (width >= breakpoints.lg) return 'lg'
    if (width >= breakpoints.md) return 'md'
    if (width >= breakpoints.sm) return 'sm'
    return 'xs'
  })

  // Device type detection
  const isMobile = computed(() => windowWidth.value < breakpoints.md)
  const isTablet = computed(() => 
    windowWidth.value >= breakpoints.md && windowWidth.value < breakpoints.lg
  )
  const isDesktop = computed(() => windowWidth.value >= breakpoints.lg)

  // Specific breakpoint checks
  const isXs = computed(() => windowWidth.value < breakpoints.sm)
  const isSm = computed(() => 
    windowWidth.value >= breakpoints.sm && windowWidth.value < breakpoints.md
  )
  const isMd = computed(() => 
    windowWidth.value >= breakpoints.md && windowWidth.value < breakpoints.lg
  )
  const isLg = computed(() => 
    windowWidth.value >= breakpoints.lg && windowWidth.value < breakpoints.xl
  )
  const isXl = computed(() => 
    windowWidth.value >= breakpoints.xl && windowWidth.value < breakpoints['2xl']
  )
  const is2Xl = computed(() => windowWidth.value >= breakpoints['2xl'])

  // Greater than or equal checks
  const smAndUp = computed(() => windowWidth.value >= breakpoints.sm)
  const mdAndUp = computed(() => windowWidth.value >= breakpoints.md)
  const lgAndUp = computed(() => windowWidth.value >= breakpoints.lg)
  const xlAndUp = computed(() => windowWidth.value >= breakpoints.xl)
  const xxlAndUp = computed(() => windowWidth.value >= breakpoints['2xl'])

  // Less than checks
  const smAndDown = computed(() => windowWidth.value < breakpoints.md)
  const mdAndDown = computed(() => windowWidth.value < breakpoints.lg)
  const lgAndDown = computed(() => windowWidth.value < breakpoints.xl)
  const xlAndDown = computed(() => windowWidth.value < breakpoints['2xl'])

  onMounted(() => {
    updateWindowWidth()
    window.addEventListener('resize', updateWindowWidth, { passive: true })
  })

  onUnmounted(() => {
    window.removeEventListener('resize', updateWindowWidth)
  })

  return {
    // Current state
    windowWidth: computed(() => windowWidth.value),
    screenSize,

    // Device types
    isMobile,
    isTablet,
    isDesktop,

    // Specific breakpoints
    isXs,
    isSm,
    isMd,
    isLg,
    isXl,
    is2Xl,

    // Greater than or equal
    smAndUp,
    mdAndUp,
    lgAndUp,
    xlAndUp,
    xxlAndUp,

    // Less than
    smAndDown,
    mdAndDown,
    lgAndDown,
    xlAndDown,

    // Breakpoint values
    breakpoints
  }
}
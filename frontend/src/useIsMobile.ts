import { useEffect, useState } from 'react'

/** Breakpoint below which the app uses the stacked mobile layout. */
export const MOBILE_QUERY = '(max-width: 768px)'

function matches(): boolean {
  // Guarded: jsdom has no matchMedia — tests exercise the desktop layout
  // unless they stub it.
  return typeof window.matchMedia === 'function' && window.matchMedia(MOBILE_QUERY).matches
}

/**
 * True on phone-sized viewports. The desktop and mobile layouts are separate
 * subtrees (not CSS-hidden duplicates), so this hook is the single switch
 * between them and follows live resizes/rotations.
 */
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(matches)

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const mql = window.matchMedia(MOBILE_QUERY)
    const onChange = (ev: { matches: boolean }) => setIsMobile(ev.matches)
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [])

  return isMobile
}

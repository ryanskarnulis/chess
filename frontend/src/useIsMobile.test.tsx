import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useIsMobile } from './useIsMobile'

type Listener = (ev: { matches: boolean }) => void

function stubMatchMedia(matches: boolean) {
  const listeners: Listener[] = []
  const mql = {
    matches,
    addEventListener: (_: string, cb: Listener) => listeners.push(cb),
    removeEventListener: (_: string, cb: Listener) => {
      const i = listeners.indexOf(cb)
      if (i >= 0) listeners.splice(i, 1)
    },
  }
  vi.stubGlobal(
    'matchMedia',
    vi.fn(() => mql),
  )
  return { listeners, mql }
}

afterEach(() => vi.unstubAllGlobals())

describe('useIsMobile', () => {
  it('reports a narrow viewport as mobile', () => {
    stubMatchMedia(true)
    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(true)
  })

  it('reports a wide viewport as desktop', () => {
    stubMatchMedia(false)
    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(false)
  })

  it('tracks breakpoint crossings', () => {
    const { listeners, mql } = stubMatchMedia(false)
    const { result } = renderHook(() => useIsMobile())
    act(() => {
      mql.matches = true
      for (const cb of listeners) cb({ matches: true })
    })
    expect(result.current).toBe(true)
  })

  it('stops listening on unmount', () => {
    const { listeners } = stubMatchMedia(false)
    const { unmount } = renderHook(() => useIsMobile())
    expect(listeners.length).toBe(1)
    unmount()
    expect(listeners.length).toBe(0)
  })

  it('defaults to desktop when matchMedia is unavailable (jsdom)', () => {
    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(false)
  })
})

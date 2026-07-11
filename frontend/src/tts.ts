// Voice out: fetch spoken audio for a piece of agent commentary and play it.
// Best-effort by design — if voice is unavailable (503/502) or autoplay is
// blocked, the commentary is still on screen, so failures are silent.
//
// Mobile autoplay: iOS Safari and Android Chrome refuse play() outside a user
// gesture, and the agent reply lands seconds after the tap that requested it.
// The escape hatch is per-element: an <audio> element that has once played
// inside a gesture may be reused programmatically forever after. So all
// playback goes through ONE shared element, and unlockAudio() — called
// synchronously from the gesture handlers — primes it with a silent clip.

// Four samples of 8 kHz 8-bit silence: the shortest well-formed WAV.
const SILENT_WAV =
  'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQQAAACAgICA'

let sharedAudio: HTMLAudioElement | null = null
let unlocked = false
/** Object URL of the clip currently loaded in the shared element. */
let liveUrl: string | null = null

function element(): HTMLAudioElement {
  if (!sharedAudio) sharedAudio = new Audio()
  return sharedAudio
}

/**
 * Prime the shared audio element so later programmatic playback is allowed.
 * MUST be called synchronously from a user-gesture handler (click/submit) —
 * that's the whole point. No-op once an unlock has succeeded.
 */
export function unlockAudio(): void {
  if (unlocked) return
  const el = element()
  el.src = SILENT_WAV
  el.play().then(
    () => {
      unlocked = true
    },
    () => {
      // Blocked even inside the gesture (or jsdom); the next gesture retries.
    },
  )
}

export async function playText(text: string): Promise<void> {
  const res = await fetch('/api/voice/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) return
  const url = URL.createObjectURL(await res.blob())
  const el = element()
  // A new clip interrupts whatever was loaded; release the old URL now and
  // guard onended so the stale clip can't revoke the live one.
  if (liveUrl) URL.revokeObjectURL(liveUrl)
  liveUrl = url
  el.src = url
  el.onended = () => {
    if (liveUrl === url) {
      URL.revokeObjectURL(url)
      liveUrl = null
    }
  }
  try {
    await el.play()
  } catch {
    // Autoplay blocked (no unlocked element) or playback failed — don't leak.
    if (liveUrl === url) {
      URL.revokeObjectURL(url)
      liveUrl = null
    }
  }
}

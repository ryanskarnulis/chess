// Voice out: fetch spoken audio for a piece of agent commentary and play it.
// Best-effort by design — if voice is unavailable (503/502) or autoplay is
// blocked, the commentary is still on screen, so failures are silent.

export async function playText(text: string): Promise<void> {
  const res = await fetch('/api/voice/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) return
  const url = URL.createObjectURL(await res.blob())
  const audio = new Audio(url)
  audio.onended = () => URL.revokeObjectURL(url)
  try {
    await audio.play()
  } catch {
    // Autoplay blocked (no user gesture) or playback failed — don't leak.
    URL.revokeObjectURL(url)
  }
}

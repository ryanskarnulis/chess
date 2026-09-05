/** The three states a copy button wears, in one place because two of them
 * live in two components now: the post-game screen's "Copy PGN" and the chip
 * under an agent reply that exported one. A copy the player cannot see the
 * result of is a copy they will make twice. */
export const COPY_IDLE = 'Copy PGN'
export const COPY_DONE = 'Copied ✓'
export const COPY_FAILED = 'Copy failed'

/**
 * Write `text` to the clipboard and report the label the button should now
 * wear. Never rejects: the clipboard is a permission away from throwing (a
 * denied prompt, an insecure origin, a browser that has none), and a failed
 * copy is something to tell the player about in the button, not an unhandled
 * rejection.
 */
export async function copyText(text: string): Promise<string> {
  try {
    await navigator.clipboard.writeText(text)
    return COPY_DONE
  } catch {
    return COPY_FAILED
  }
}

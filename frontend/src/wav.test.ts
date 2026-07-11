import { describe, expect, it } from 'vitest'
import { encodeWav } from './wav'

// The VAD hands utterances over as Float32Array samples (16 kHz mono); the
// transcribe endpoint wants a real audio container. This encoder is the
// bridge: 16-bit PCM WAV, the simplest format every STT backend accepts.

async function bytes(blob: Blob): Promise<DataView> {
  return new DataView(await blob.arrayBuffer())
}

function ascii(view: DataView, offset: number, length: number): string {
  let s = ''
  for (let i = 0; i < length; i++) s += String.fromCharCode(view.getUint8(offset + i))
  return s
}

describe('encodeWav', () => {
  it('produces a well-formed 16 kHz mono 16-bit PCM WAV header', async () => {
    const blob = encodeWav(new Float32Array([0, 0.5, -0.5, 1]))
    expect(blob.type).toBe('audio/wav')
    const view = await bytes(blob)

    expect(ascii(view, 0, 4)).toBe('RIFF')
    // RIFF size = total - 8; header is 44 bytes, 4 samples × 2 bytes = 8.
    expect(view.getUint32(4, true)).toBe(44 + 8 - 8)
    expect(ascii(view, 8, 4)).toBe('WAVE')
    expect(ascii(view, 12, 4)).toBe('fmt ')
    expect(view.getUint32(16, true)).toBe(16) // fmt chunk size
    expect(view.getUint16(20, true)).toBe(1) // PCM
    expect(view.getUint16(22, true)).toBe(1) // mono
    expect(view.getUint32(24, true)).toBe(16000) // sample rate
    expect(view.getUint32(28, true)).toBe(32000) // byte rate = rate × 2
    expect(view.getUint16(32, true)).toBe(2) // block align
    expect(view.getUint16(34, true)).toBe(16) // bits per sample
    expect(ascii(view, 36, 4)).toBe('data')
    expect(view.getUint32(40, true)).toBe(8) // data bytes
  })

  it('scales samples to signed 16-bit integers', async () => {
    const blob = encodeWav(new Float32Array([0, 1, -1, 0.5]))
    const view = await bytes(blob)
    expect(view.getInt16(44, true)).toBe(0)
    expect(view.getInt16(46, true)).toBe(32767)
    expect(view.getInt16(48, true)).toBe(-32768)
    expect(view.getInt16(50, true)).toBe(Math.round(0.5 * 32767))
  })

  it('clips out-of-range samples instead of wrapping around', async () => {
    const blob = encodeWav(new Float32Array([1.5, -1.5]))
    const view = await bytes(blob)
    expect(view.getInt16(44, true)).toBe(32767)
    expect(view.getInt16(46, true)).toBe(-32768)
  })
})

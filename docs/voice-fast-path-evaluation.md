# Web Speech API fast path vs local-only voice — evaluation

**Decision: local-only (Speaches). No Web Speech API path. (2026-07-06)**

The last open question of the Voice epic (BRIEF: "browser Web Speech API is
the fast path but some browsers send audio to the cloud; prefer local Whisper
if strict privacy is required").

## The contenders

- **Web Speech API (`SpeechRecognition`)** — built into the browser, zero
  backend work, near-instant interim results.
- **Local-only** — the shipped path: browser `MediaRecorder` →
  `/api/voice/transcribe` → Speaches (faster-whisper) on the home box.

## Why local-only wins

1. **Privacy/offline is a project invariant, not a preference.** The brief
   pins "basic gameplay must work fully offline / no public internet. Fast and
   private." Chrome and Safari implement `SpeechRecognition` by shipping the
   audio to Google/Apple servers — the mic stream of a private home app would
   leave the LAN. That's not a fast path, it's a different product.
2. **It doesn't even work where it matters.** Firefox has no usable
   `SpeechRecognition`; Chrome's requires internet — so the "fast path" fails
   exactly when the offline guarantee is being exercised, and a second STT
   path would still be needed as the fallback. Two paths, one of which
   violates the invariant, versus one path that always works.
3. **Measured local latency is fine for chess.** Round-trip through the real
   stack (Kokoro-spoken ~3s command "Knight to f3, then castle kingside" →
   Speaches `Systran/faster-whisper-small`, CPU-only, llama holding the GPU):
   **~2.5s warm** (2.5–3.1s over three runs) with a **perfect chess-vocabulary
   transcription**. Chess is turn-based; 2.5s between speaking and the agent
   acting is comfortably inside the thinking rhythm of a move.
4. **One road in.** The architecture principle is a single pipeline (input →
   string → agent). The mic path already joins it at `/api/voice/transcribe` →
   `/api/command`. A browser-side recognizer would be a second, differently
   behaving transcriber feeding the same pipe — more surface, no new
   capability.

## What we give up, and the escape hatches

- Interim ("live") transcription while speaking. Not needed for short
  imperative commands; if wanted later, Speaches exposes a realtime WebSocket
  API — still local.
- If 2.5s ever feels slow: `Systran/faster-distil-whisper-small.en` or
  `tiny.en` (roughly 2–4× faster, English-only) is a one-line
  `STT_MODEL` change; or give Speaches the GPU tag on a box with
  spare VRAM. Both stay local; neither touches architecture.

Web Speech *synthesis* (TTS) was considered too: it is on-device in most
browsers, but voice quality is erratic across devices and we already have
Kokoro through the same server — one voice, everywhere, also local. Same
verdict.

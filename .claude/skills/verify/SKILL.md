---
name: verify
description: How to run and drive this app to verify frontend/backend changes end-to-end on this machine.
---

# Verifying a change in the running app

## Handles

- The full stack usually already runs in Docker: `chess-app-1` (app, http://localhost:8000) plus two external shared stacks — the brain (`../llama-swap/`, host :8200) and voice (`../speech/`: `speech-speaches-1` host :8400, `speech-kokoro-1` host :8410). Check with `docker ps`. The container serves the **last built image** — it will not reflect working-tree changes.
- To drive working-tree frontend code: `cd frontend && npm run dev` (background) — Vite proxies `/api` and `/ws` to localhost:8000. **Check the log for the actual port**: it takes 5174+ if 5173 is busy.
- Driving the shared backend touches the user's live game state — stick to read-only/chat commands unless the change needs moves.

## Browser driving

- No system Chrome. Playwright's Chromium is cached at `~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome`; use `playwright-core` with `executablePath` pointing there.
- **WebKit does not run on this Fedora host** (Ubuntu-built, missing sonames). Use the official image with host networking instead:
  `docker run --rm --network host -v <scriptdir>:/work -w /work -e PLAYWRIGHT_BROWSERS_PATH=/ms-playwright mcr.microsoft.com/playwright:v<pw-version>-noble node <script>.mjs`
  (image already pulled for v1.61.1).
- Useful selectors: `input[aria-label="Command"]`, `button[type="submit"]`, `button.voice-toggle`, `.commentary` / `.commentary-thinking`. Agent replies take up to ~1–2 min (local LLM) — wait for `.commentary:not(.commentary-thinking)` with a long timeout.

## Gotchas

- Playwright browsers (Chromium and WebKit alike) grant sticky autoplay activation after any click — they **cannot reproduce mobile autoplay blocking** (iOS transient-gesture rules). Verify audio-unlock behavior by instrumenting `HTMLMediaElement.prototype.play` in `addInitScript` and asserting the *sequence* (unlock clip inside the gesture, playback on the same element); final confirmation needs a real phone.
- **Chromium's fake mic is broken here** — `--use-fake-device-for-media-capture` (± `--use-file-for-fake-audio-capture`) yields `NotReadableError` on the host and `NotFoundError` (zero devices) in the Playwright image, headless or not. To feed audio into getUserMedia (e.g. driving the hands-free VAD), patch it in `addInitScript`: decode a base64-embedded WAV with `AudioContext.decodeAudioData`, play it looped through `createMediaStreamDestination()`, and return that stream. Generate the clip with the stack's own TTS (`POST /api/voice/speak`, returns MP3 — `ffmpeg -i clip.mp3 -af "apad=pad_dur=4" -ar 16000 -ac 1 clip.wav`; the silence tail is what lets the VAD endpoint the loop). Grant `permissions: ['microphone']` on the context. Everything past capture — worklet, WASM VAD, STT, agent — runs for real; see the hands-free states via `button.mic-listening` / `.mic-capturing` and `button[aria-label="Stop listening"]`.

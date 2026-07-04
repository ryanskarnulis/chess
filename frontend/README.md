# Chess — frontend

The web board for the chess app. React + TypeScript + Vite, with
[Chessground](https://github.com/lichess-org/chessground) (Lichess's board) as
the rendering layer.

The board is display-only for now: the backend (`../backend`) is the single
source of truth for game state and legality. Later slices wire this to the
backend's state channel and route moves through it.

## Commands

```bash
npm install        # install deps
npm run dev        # dev server (http://localhost:5173)
npm run build      # type-check (tsc) + production build
npm run lint       # oxlint
npm test           # vitest (jsdom)
npm run test:watch # vitest in watch mode
```

## Layout

- `src/Board.tsx` — thin React wrapper owning a Chessground instance's lifecycle.
- `src/App.tsx` — app shell; renders the board at the starting position.
- `vite.config.ts` — build config. `vitest.config.ts` — test config (kept
  separate so the build's `tsc -b` never reconciles Vitest's bundled Vite types
  with the app's Vite).

/**
 * The app's inline icon set.
 *
 * These replace the emoji glyphs the controls used to carry. On iOS an
 * emoji-presentation character is rendered by Apple Color Emoji whatever the
 * surrounding style says, so `↩` came out as the blue ↩️ tile, the flag and
 * bulb came out in full color, and the monochrome silk chrome had four
 * differently-styled pictures in it. A character class can't be styled; a
 * path can.
 *
 * Every icon draws in `currentColor`, so a control's own color and its
 * disabled/hover states carry the icon with them, and every one is
 * `aria-hidden` — the button's visible label or `aria-label` is the
 * accessible name, and an icon announced alongside it would only repeat.
 */

interface IconProps {
  /** Square edge in px. Defaults to the 22px the control rows use. */
  size?: number
}

function Icon({ size = 22, children }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  )
}

export function MenuIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 7h16M4 12h16M4 17h16" />
    </Icon>
  )
}

export function FlagIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M6 21V4" />
      <path d="M6 4h11l-2 3.5L17 11H6" />
    </Icon>
  )
}

export function BulbIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M9.5 17h5" />
      <path d="M10 20h4" />
      <path d="M12 3a6 6 0 0 0-3.5 10.9c.4.3.5.7.5 1.1h6c0-.4.1-.8.5-1.1A6 6 0 0 0 12 3Z" />
    </Icon>
  )
}

export function UndoIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 9h11a5 5 0 0 1 0 10h-6" />
      <path d="m8 5-4 4 4 4" />
    </Icon>
  )
}

export function MicIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </Icon>
  )
}

/** Listening: the mic with sound arriving, so the state reads at a glance. */
export function ListeningIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
      <path d="M2.5 8.5v3M21.5 8.5v3" />
    </Icon>
  )
}

export function SpeakerOnIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 9.5h3L11 6v12l-4-3.5H4Z" />
      <path d="M15 9.5a3.5 3.5 0 0 1 0 5" />
      <path d="M17.5 7a7 7 0 0 1 0 10" />
    </Icon>
  )
}

export function SpeakerOffIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 9.5h3L11 6v12l-4-3.5H4Z" />
      <path d="m15.5 9.5 5 5M20.5 9.5l-5 5" />
    </Icon>
  )
}

export function ChevronLeftIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m14.5 6-5.5 6 5.5 6" />
    </Icon>
  )
}

export function ChevronRightIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m9.5 6 5.5 6-5.5 6" />
    </Icon>
  )
}

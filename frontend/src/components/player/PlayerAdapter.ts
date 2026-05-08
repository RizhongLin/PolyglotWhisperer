/**
 * Provider-agnostic interface the transcript uses to drive playback.
 *
 * Each provider (HTML5, YouTube, Vimeo, Unavailable) implements this
 * tiny surface and the route only needs to know which adapter to mount;
 * it does not care about iframe APIs or `<video>` ref details.
 *
 * Time is in seconds throughout. Subscription returns an unsubscribe
 * function — the transcript drives a 4Hz progress poller off this so
 * we don't depend on a particular provider's tick cadence.
 */
export interface PlayerAdapter {
  /** Current playhead in seconds. May be 0 before the player is ready. */
  getCurrentTime(): number;
  /** Total duration in seconds. May be 0 before metadata loads. */
  getDuration(): number;
  /** Seek to ``seconds``. No-op if the underlying player isn't ready. */
  seek(seconds: number): void;
  /** Subscribe to time updates. Adapters fire at provider cadence (≥4Hz). */
  onTime(handler: (seconds: number) => void): () => void;
  /** Subscribe to readiness — fires once when ``getDuration() > 0``. */
  onReady(handler: () => void): () => void;
}

/** Marker constant for the route ↔ adapter contract — easier to grep. */
export const ADAPTER_TIME_HZ = 4;

import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import type { PlayerAdapter } from './PlayerAdapter';

interface YoutubePlayerProps {
  videoId: string;
  className?: string;
  /** Called when the YouTube iframe refuses to load (X-Frame-Options). */
  onEmbedRefused?: () => void;
}

interface YTPlayer {
  getCurrentTime(): number;
  getDuration(): number;
  seekTo(seconds: number, allowSeekAhead: boolean): void;
}

declare global {
  interface Window {
    onYouTubeIframeAPIReady?: () => void;
    YT?: {
      Player: new (
        el: HTMLElement,
        opts: {
          videoId: string;
          events?: {
            onReady?: (e: { target: YTPlayer }) => void;
            onError?: (e: { data: number }) => void;
          };
          playerVars?: Record<string, string | number>;
        },
      ) => YTPlayer;
    };
  }
}

const YT_API_SRC = 'https://www.youtube.com/iframe_api';
let ytApiLoading: Promise<void> | null = null;

function loadIframeApi(): Promise<void> {
  if (window.YT?.Player) return Promise.resolve();
  if (ytApiLoading) return ytApiLoading;
  ytApiLoading = new Promise<void>((resolve) => {
    const prior = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      prior?.();
      resolve();
    };
    const tag = document.createElement('script');
    tag.src = YT_API_SRC;
    tag.async = true;
    document.head.appendChild(tag);
  });
  return ytApiLoading;
}

/**
 * YouTube IFrame API behind the same ``PlayerAdapter`` shape.
 *
 * Polls ``getCurrentTime`` at 4Hz because the IFrame API doesn't emit
 * a ``timeupdate``-style event — same cadence as a sluggish HTML5
 * player, plenty for cue highlighting.
 */
export const YoutubePlayer = forwardRef<PlayerAdapter, YoutubePlayerProps>(function YoutubePlayer(
  { videoId, className, onEmbedRefused },
  ref,
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<YTPlayer | null>(null);
  const timeHandlersRef = useRef<Set<(s: number) => void>>(new Set());
  const readyHandlersRef = useRef<Set<() => void>>(new Set());
  const readyFiredRef = useRef(false);

  useImperativeHandle(
    ref,
    (): PlayerAdapter => ({
      getCurrentTime: () => playerRef.current?.getCurrentTime() ?? 0,
      getDuration: () => playerRef.current?.getDuration() ?? 0,
      seek: (s: number) => playerRef.current?.seekTo(s, true),
      onTime: (h) => {
        timeHandlersRef.current.add(h);
        return () => timeHandlersRef.current.delete(h);
      },
      onReady: (h) => {
        if (readyFiredRef.current) {
          h();
          return () => undefined;
        }
        readyHandlersRef.current.add(h);
        return () => readyHandlersRef.current.delete(h);
      },
    }),
    [],
  );

  useEffect(() => {
    let cancelled = false;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    void loadIframeApi().then(() => {
      if (cancelled || !containerRef.current || !window.YT?.Player) return;
      // YT.Player replaces the target element in-place, so we mount it
      // on a child div we can re-create on unmount.
      const target = document.createElement('div');
      containerRef.current.replaceChildren(target);
      playerRef.current = new window.YT.Player(target, {
        videoId,
        playerVars: { rel: 0, modestbranding: 1, playsinline: 1 },
        events: {
          onReady: ({ target: p }) => {
            readyFiredRef.current = true;
            readyHandlersRef.current.forEach((h) => h());
            readyHandlersRef.current.clear();
            pollTimer = setInterval(() => {
              const t = p.getCurrentTime();
              timeHandlersRef.current.forEach((h) => h(t));
            }, 250);
          },
          onError: (e) => {
            // 101/150 = embedding disabled by uploader; 5 = HTML5 player error.
            if ([101, 150, 5].includes(e.data)) onEmbedRefused?.();
          },
        },
      });
    });

    return () => {
      cancelled = true;
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [videoId, onEmbedRefused]);

  return <div ref={containerRef} className={className} />;
});

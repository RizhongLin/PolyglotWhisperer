import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import type { PlayerAdapter } from './PlayerAdapter';

interface VimeoPlayerProps {
  videoId: string;
  className?: string;
  onEmbedRefused?: () => void;
}

interface VimeoApi {
  getCurrentTime(): Promise<number>;
  getDuration(): Promise<number>;
  setCurrentTime(seconds: number): Promise<number>;
  on(event: string, handler: (data: { seconds: number; duration?: number }) => void): void;
  ready(): Promise<void>;
}

declare global {
  interface Window {
    Vimeo?: {
      Player: new (el: HTMLElement, opts: { id: string }) => VimeoApi;
    };
  }
}

const VIMEO_API_SRC = 'https://player.vimeo.com/api/player.js';
let vimeoLoading: Promise<void> | null = null;

function loadVimeoApi(): Promise<void> {
  if (window.Vimeo?.Player) return Promise.resolve();
  if (vimeoLoading) return vimeoLoading;
  vimeoLoading = new Promise<void>((resolve, reject) => {
    const tag = document.createElement('script');
    tag.src = VIMEO_API_SRC;
    tag.async = true;
    tag.onload = () => resolve();
    tag.onerror = () => reject(new Error('Vimeo player.js failed to load'));
    document.head.appendChild(tag);
  });
  return vimeoLoading;
}

/** Vimeo embed driven through ``PlayerAdapter``. */
export const VimeoPlayer = forwardRef<PlayerAdapter, VimeoPlayerProps>(function VimeoPlayer(
  { videoId, className, onEmbedRefused },
  ref,
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<VimeoApi | null>(null);
  const lastTimeRef = useRef(0);
  const lastDurationRef = useRef(0);
  const timeHandlersRef = useRef<Set<(s: number) => void>>(new Set());
  const readyHandlersRef = useRef<Set<() => void>>(new Set());
  const readyFiredRef = useRef(false);

  useImperativeHandle(
    ref,
    (): PlayerAdapter => ({
      getCurrentTime: () => lastTimeRef.current,
      getDuration: () => lastDurationRef.current,
      seek: (s: number) => {
        playerRef.current?.setCurrentTime(s).catch(() => undefined);
      },
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

    loadVimeoApi()
      .then(() => {
        if (cancelled || !containerRef.current || !window.Vimeo?.Player) return;
        const target = document.createElement('div');
        containerRef.current.replaceChildren(target);
        const player = new window.Vimeo.Player(target, { id: videoId });
        playerRef.current = player;
        player
          .ready()
          .then(() => player.getDuration())
          .then((d) => {
            lastDurationRef.current = d;
            readyFiredRef.current = true;
            readyHandlersRef.current.forEach((h) => h());
            readyHandlersRef.current.clear();
          })
          .catch(() => onEmbedRefused?.());
        player.on('timeupdate', ({ seconds, duration }) => {
          lastTimeRef.current = seconds;
          if (duration) lastDurationRef.current = duration;
          timeHandlersRef.current.forEach((h) => h(seconds));
        });
      })
      .catch(() => onEmbedRefused?.());

    return () => {
      cancelled = true;
    };
  }, [videoId, onEmbedRefused]);

  return <div ref={containerRef} className={className} />;
});

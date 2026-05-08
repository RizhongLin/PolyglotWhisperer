import { forwardRef, useImperativeHandle, useRef } from 'react';
import { TriangleAlert } from 'lucide-react';
import type { PlayerAdapter } from './PlayerAdapter';

interface UnavailablePlayerProps {
  className?: string;
  reason?: string;
  /** Optional re-download trigger surfaced as a button. */
  onRedownload?: () => void;
}

/**
 * Placeholder rendered when the video file is missing AND no embed is
 * available. The transcript still works — it polls a stub adapter that
 * keeps reporting time 0, which is fine for click-to-seek (the seeks
 * are no-ops; the transcript reads as a static document).
 */
export const UnavailablePlayer = forwardRef<PlayerAdapter, UnavailablePlayerProps>(
  function UnavailablePlayer({ className, reason, onRedownload }, ref) {
    const handlersRef = useRef<Set<() => void>>(new Set());
    useImperativeHandle(
      ref,
      (): PlayerAdapter => ({
        getCurrentTime: () => 0,
        getDuration: () => 0,
        seek: () => undefined,
        onTime: () => () => undefined,
        onReady: (h) => {
          handlersRef.current.add(h);
          h();
          return () => handlersRef.current.delete(h);
        },
      }),
      [],
    );

    return (
      <div
        className={
          className ??
          'flex aspect-video flex-col items-center justify-center gap-3 rounded-md border bg-muted/30 p-6 text-center'
        }
      >
        <TriangleAlert className="size-8 text-muted-foreground" />
        <p className="text-sm font-medium">Video unavailable</p>
        <p className="max-w-md text-xs text-muted-foreground">
          {reason ?? 'The video file is not present and no provider embed is available.'}
        </p>
        {onRedownload ? (
          <button
            type="button"
            onClick={onRedownload}
            className="mt-2 rounded-md border bg-background px-3 py-1.5 text-xs font-medium hover:bg-accent"
          >
            Re-download
          </button>
        ) : null}
      </div>
    );
  },
);

import { useEffect, useMemo, useRef, useState } from 'react';
import { Volume2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { RatingButtons } from './RatingButtons';
import type { FlashcardResponse, FsrsRating } from '@/api/types';

interface ReviewCardProps {
  card: FlashcardResponse;
  audioUrl: string | null;
  onRate: (rating: FsrsRating, elapsedMs: number) => void;
  busy: boolean;
}

/**
 * One review session's worth of UI: front → reveal → grade.
 *
 * Audio (when present) auto-plays once on mount and is replayable via
 * the speaker button. We track elapsed ms from card mount to grade so
 * the FSRS log captures study time.
 */
export function ReviewCard({ card, audioUrl, onRate, busy }: ReviewCardProps) {
  const [revealed, setRevealed] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mountedAt = useMemo(() => Date.now(), [card.id]);

  // Reset reveal state when the card changes.
  useEffect(() => setRevealed(false), [card.id]);

  // Auto-play audio on mount.
  useEffect(() => {
    if (!audioUrl) return;
    const a = audioRef.current;
    if (!a) return;
    a.play().catch(() => undefined);
  }, [audioUrl, card.id]);

  const onSpeaker = () => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = 0;
    void a.play();
  };

  const handleRate = (rating: FsrsRating) => {
    onRate(rating, Date.now() - mountedAt);
  };

  return (
    <Card className="flex flex-col gap-4 p-6">
      <div className="flex items-start justify-between gap-3">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
          {card.language}
        </span>
        {audioUrl ? (
          <button
            type="button"
            onClick={onSpeaker}
            className="rounded-md border bg-background p-1.5 hover:bg-accent"
            aria-label="Replay audio"
          >
            <Volume2 className="size-4" />
          </button>
        ) : null}
      </div>

      <div className="text-center text-2xl font-semibold leading-snug">{card.front}</div>

      {audioUrl ? (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <audio ref={audioRef} src={audioUrl} preload="auto" />
      ) : null}

      {revealed ? (
        <>
          <div className="rounded-md bg-muted/50 p-4 text-center text-base">{card.back}</div>
          <RatingButtons onRate={handleRate} disabled={busy} />
        </>
      ) : (
        <button
          type="button"
          onClick={() => setRevealed(true)}
          className="rounded-md border bg-background py-3 text-sm font-medium hover:bg-accent"
        >
          Show answer
        </button>
      )}
    </Card>
  );
}

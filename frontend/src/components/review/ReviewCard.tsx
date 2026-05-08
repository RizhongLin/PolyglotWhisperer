import { useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, Sparkles, Trash2, Volume2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { RatingButtons } from './RatingButtons';
import type { FlashcardResponse, FsrsRating } from '@/api/types';

interface ReviewCardProps {
  card: FlashcardResponse;
  audioUrl: string | null;
  onRate: (rating: FsrsRating, elapsedMs: number) => void;
  onDiscard: () => void;
  busy: boolean;
}

/**
 * One review session's worth of UI: front → reveal → grade.
 *
 * Renders the LLM-refined fields when ``refine_status === 'done'``,
 * otherwise falls back to the original ``front``/``back``. Audio
 * (when present) auto-plays once on mount and is replayable via the
 * speaker button. We track elapsed ms from card mount to grade so the
 * FSRS log captures study time. Discard removes the card outright —
 * useful for noise the LLM can't fix.
 */
export function ReviewCard({ card, audioUrl, onRate, onDiscard, busy }: ReviewCardProps) {
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

  const refined = card.refine_status === 'done' && card.definition;

  return (
    <Card className="flex flex-col gap-4 p-6">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>{card.language}</span>
          {card.pos ? (
            <span className="rounded-sm bg-muted px-1 py-px font-mono">{card.pos}</span>
          ) : null}
          {card.refine_status === 'pending' ? (
            <span className="inline-flex items-center gap-1 text-amber-500">
              <Loader2 className="size-3 animate-spin" /> refining
            </span>
          ) : null}
          {refined ? (
            <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
              <Sparkles className="size-3" /> refined
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-1">
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
          <button
            type="button"
            onClick={onDiscard}
            disabled={busy}
            className="rounded-md border bg-background p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
            aria-label="Discard card"
            title="Discard this card permanently"
          >
            <Trash2 className="size-4" />
          </button>
        </div>
      </div>

      <div className="text-center text-2xl font-semibold leading-snug">
        {refined && card.lemma ? card.lemma : card.front}
      </div>

      {audioUrl ? (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <audio ref={audioRef} src={audioUrl} preload="auto" />
      ) : null}

      {revealed ? (
        <>
          <div className="flex flex-col gap-2 rounded-md bg-muted/50 p-4 text-center">
            <div className="text-base font-medium">
              {refined && card.definition ? card.definition : card.back}
            </div>
            {refined && card.example_source ? (
              <div className="border-t border-border/50 pt-2 text-xs">
                <div className="italic">{card.example_source}</div>
                {card.example_target ? (
                  <div className="text-muted-foreground">{card.example_target}</div>
                ) : null}
              </div>
            ) : null}
            {refined && card.mnemonic ? (
              <div className="border-t border-border/50 pt-2 text-[11px] text-muted-foreground">
                💡 {card.mnemonic}
              </div>
            ) : null}
          </div>
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

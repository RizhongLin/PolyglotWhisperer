import { useEffect, useMemo, useState } from 'react';
import { Link, createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { Brain, CheckCircle2, Inbox, Loader2 } from 'lucide-react';
import { ApiError, api } from '@/api/client';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ReviewCard } from '@/components/review/ReviewCard';
import type { FsrsRating } from '@/api/types';

export const Route = createFileRoute('/review')({
  component: ReviewPage,
});

function ReviewPage() {
  const queue = useQuery({
    queryKey: ['flashcard-queue'],
    queryFn: () => api.flashcardQueue(50),
    staleTime: 0,
  });

  const [index, setIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reviewed, setReviewed] = useState(0);

  // Reset position when the queue refetches with a different first id.
  useEffect(() => {
    setIndex(0);
  }, [queue.dataUpdatedAt]);

  const cards = queue.data ?? [];
  const card = cards[index];

  // Poll every 4 s while the current card is still being LLM-refined so
  // the enriched fields appear without the user having to refresh.
  // Stops as soon as the card flips to a terminal status.
  useEffect(() => {
    if (!card || card.refine_status !== 'pending') return;
    const handle = setInterval(() => {
      void queue.refetch();
    }, 4_000);
    return () => clearInterval(handle);
  }, [card?.id, card?.refine_status, queue]);

  const audioUrl = useMemo(() => {
    if (!card) return null;
    if (card.audio_start_ms == null || card.audio_end_ms == null) return null;
    return api.audioClipUrl(
      card.workspace_slug,
      card.workspace_timestamp,
      card.audio_start_ms,
      card.audio_end_ms,
    );
  }, [card]);

  const advance = async () => {
    if (index + 1 >= cards.length) {
      await queue.refetch();
    } else {
      setIndex(index + 1);
    }
  };

  const onRate = async (rating: FsrsRating, elapsedMs: number) => {
    if (!card) return;
    setBusy(true);
    setError(null);
    try {
      await api.reviewFlashcard(card.id, rating, elapsedMs);
      setReviewed((n) => n + 1);
      await advance();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onDiscard = async () => {
    if (!card) return;
    if (!confirm('Discard this card permanently?')) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteFlashcard(card.id);
      await advance();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (queue.isLoading) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-muted-foreground">
        <Loader2 className="mr-2 size-4 animate-spin" /> Loading queue…
      </div>
    );
  }

  if (queue.isError) {
    return (
      <Card className="p-6">
        <p className="text-sm text-destructive">
          Failed to load review queue: {(queue.error as Error).message}
        </p>
      </Card>
    );
  }

  return (
    <div className="mx-auto flex max-w-md flex-col gap-6">
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary/10">
            <Brain className="size-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Review</h1>
            <p className="text-sm text-muted-foreground">
              Spaced-repetition flashcard review
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="tabular-nums">
            {reviewed} done
          </Badge>
          <Badge variant="outline" className="tabular-nums">
            {Math.max(0, cards.length - index)} left
          </Badge>
        </div>
      </section>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      {!card ? (
        <Card className="flex flex-col items-center gap-4 p-10 text-center">
          {reviewed > 0 ? (
            <>
              <div className="flex size-12 items-center justify-center rounded-xl bg-emerald-500/10">
                <CheckCircle2 className="size-6 text-emerald-500" />
              </div>
              <p className="text-sm">Queue empty — {reviewed} cards reviewed.</p>
            </>
          ) : (
            <>
              <div className="flex size-12 items-center justify-center rounded-xl bg-muted">
                <Inbox className="size-6 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">No cards due right now.</p>
            </>
          )}
          <Link to="/library" className="text-xs text-primary hover:underline">
            Back to library
          </Link>
        </Card>
      ) : (
        <ReviewCard
          card={card}
          audioUrl={audioUrl}
          onRate={onRate}
          onDiscard={onDiscard}
          busy={busy}
        />
      )}
    </div>
  );
}


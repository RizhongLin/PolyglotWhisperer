import { useEffect, useMemo, useState } from 'react';
import { Link, createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { CheckCircle2, Inbox, Loader2 } from 'lucide-react';
import { ApiError, api } from '@/api/client';
import { Card } from '@/components/ui/card';
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

  const onRate = async (rating: FsrsRating, elapsedMs: number) => {
    if (!card) return;
    setBusy(true);
    setError(null);
    try {
      await api.reviewFlashcard(card.id, rating, elapsedMs);
      setReviewed((n) => n + 1);
      if (index + 1 >= cards.length) {
        await queue.refetch();
      } else {
        setIndex(index + 1);
      }
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
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Review</h1>
        <span className="text-xs text-muted-foreground tabular-nums">
          {reviewed} done · {Math.max(0, cards.length - index)} left
        </span>
      </header>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      {!card ? (
        <Card className="flex flex-col items-center gap-3 p-10 text-center">
          {reviewed > 0 ? (
            <>
              <CheckCircle2 className="size-10 text-emerald-500" />
              <p className="text-sm">Queue empty — {reviewed} cards reviewed.</p>
            </>
          ) : (
            <>
              <Inbox className="size-10 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No cards due right now.</p>
            </>
          )}
          <Link to="/library" className="text-xs text-primary hover:underline">
            Back to library
          </Link>
        </Card>
      ) : (
        <ReviewCard card={card} audioUrl={audioUrl} onRate={onRate} busy={busy} />
      )}
    </div>
  );
}


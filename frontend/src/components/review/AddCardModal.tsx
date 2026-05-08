import { useEffect, useRef, useState } from 'react';
import { Loader2, Plus, X } from 'lucide-react';
import { ApiError, api } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { CreateFlashcardRequest } from '@/api/types';

interface AddCardModalProps {
  workspaceId: number;
  language: string;
  /** Initial values — typically the clicked word + its translation/context. */
  initialFront?: string;
  initialBack?: string;
  initialAudioStartMs?: number;
  initialAudioEndMs?: number;
  open: boolean;
  onClose: () => void;
  onCreated?: () => void;
}

/**
 * Lightweight modal — no Radix dialog dep here, just a fixed-position
 * card with backdrop. The transcript-side popover triggers it pre-
 * filled; the user can edit before submitting.
 */
export function AddCardModal({
  workspaceId,
  language,
  initialFront,
  initialBack,
  initialAudioStartMs,
  initialAudioEndMs,
  open,
  onClose,
  onCreated,
}: AddCardModalProps) {
  const [front, setFront] = useState(initialFront ?? '');
  const [back, setBack] = useState(initialBack ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const frontRef = useRef<HTMLInputElement>(null);

  // Re-seed when re-opened with different defaults.
  useEffect(() => {
    if (open) {
      setFront(initialFront ?? '');
      setBack(initialBack ?? '');
      setError(null);
      setTimeout(() => frontRef.current?.focus(), 50);
    }
  }, [open, initialFront, initialBack]);

  if (!open) return null;

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const payload: CreateFlashcardRequest = {
        workspace_id: workspaceId,
        front: front.trim(),
        back: back.trim(),
        language,
        ...(initialAudioStartMs != null && initialAudioEndMs != null
          ? { audio_start_ms: initialAudioStartMs, audio_end_ms: initialAudioEndMs }
          : {}),
      };
      await api.createFlashcard(payload);
      onCreated?.();
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <form
        onSubmit={onSubmit}
        onClick={(e) => e.stopPropagation()}
        className="flex w-full max-w-md flex-col gap-4 rounded-lg border bg-card p-6 shadow-lg"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">New flashcard</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground hover:bg-accent"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="card-front">Front (prompt)</Label>
          <Input
            ref={frontRef}
            id="card-front"
            value={front}
            onChange={(e) => setFront(e.target.value)}
            required
            maxLength={1024}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="card-back">Back (answer)</Label>
          <Input
            id="card-back"
            value={back}
            onChange={(e) => setBack(e.target.value)}
            required
            maxLength={4096}
          />
        </div>

        {initialAudioStartMs != null && initialAudioEndMs != null ? (
          <p className="text-[11px] text-muted-foreground">
            Audio: {(initialAudioStartMs / 1000).toFixed(1)}s –{' '}
            {(initialAudioEndMs / 1000).toFixed(1)}s
          </p>
        ) : null}

        {error ? <p className="text-xs text-destructive">{error}</p> : null}

        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button type="submit" disabled={busy || !front.trim() || !back.trim()}>
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
            Save
          </Button>
        </div>
      </form>
    </div>
  );
}

import { useEffect, useMemo } from 'react';
import { Link } from '@tanstack/react-router';
import { Activity, CheckCircle2, X } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button, buttonClass } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { api, openJobStream } from '@/api/client';
import type { JobEvent, JobRecord, JobState } from '@/api/types';
import { formatStage } from '@/lib/format';

const STATE_LABEL: Record<JobState, string> = {
  pending: 'Queued',
  running: 'Running',
  cancelling: 'Cancelling…',
  cancelled: 'Cancelled',
  succeeded: 'Done',
  failed: 'Failed',
  interrupted: 'Interrupted',
};

const STATE_BADGE: Record<JobState, 'default' | 'secondary' | 'success' | 'warning' | 'destructive'> = {
  pending: 'secondary',
  running: 'default',
  cancelling: 'warning',
  cancelled: 'warning',
  succeeded: 'success',
  failed: 'destructive',
  interrupted: 'destructive',
};

export const TERMINAL_STATES: ReadonlySet<JobState> = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
]);

interface JobsStripProps {
  jobs: JobRecord[];
  onPatch: (id: string, patch: Partial<JobRecord>) => void;
  onTerminal: (id: string) => void;
  onDismiss: (id: string) => void;
}

export function JobsStrip({ jobs, onPatch, onTerminal, onDismiss }: JobsStripProps) {
  if (jobs.length === 0) return null;
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        <Activity className="mr-1 inline size-3.5" />
        Jobs
      </h2>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {jobs.map((job) => (
          <JobCard
            key={job.id}
            job={job}
            onPatch={(patch) => onPatch(job.id, patch)}
            onTerminal={() => onTerminal(job.id)}
            onDismiss={() => onDismiss(job.id)}
          />
        ))}
      </div>
    </section>
  );
}

interface JobCardProps {
  job: JobRecord;
  onPatch: (patch: Partial<JobRecord>) => void;
  onTerminal: () => void;
  onDismiss: () => void;
}

function JobCard({ job, onPatch, onTerminal, onDismiss }: JobCardProps) {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let close: () => void;

    const connect = () => {
      close = openJobStream(
        job.id,
        (ev) => applyEvent(ev, onPatch, onTerminal),
        (err) => {
          console.warn('stream error', err);
          timer = setTimeout(connect, 2_000);
        },
      );
    };
    connect();

    return () => {
      close?.();
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  const target = useMemo(() => {
    const lang = job.inputs.language;
    const tgt = job.inputs.translate;
    return tgt ? `${lang} → ${tgt}` : lang;
  }, [job.inputs.language, job.inputs.translate]);

  const cancel = async () => {
    try {
      await api.cancelJob(job.id);
    } catch (err) {
      console.warn('cancel failed', err);
    }
  };

  const isTerminal = TERMINAL_STATES.has(job.state);
  const wsLink =
    job.slug && job.timestamp
      ? { to: '/library/$slug/$ts' as const, params: { slug: job.slug, ts: job.timestamp } }
      : null;

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="line-clamp-1 text-sm font-medium" title={job.inputs.input}>
            {job.inputs.input}
          </div>
          <div className="text-xs text-muted-foreground">{target}</div>
        </div>
        <Badge variant={STATE_BADGE[job.state]}>{STATE_LABEL[job.state] ?? job.state}</Badge>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <Progress value={job.progress} className="flex-1" />
        <span className="w-12 text-right text-xs text-muted-foreground tabular-nums">
          {Math.round((job.progress || 0) * 100)}%
        </span>
      </div>

      <div className="mt-2 flex items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>{formatStage(job.stage)}</span>
        <span className="line-clamp-1 text-right">{job.message ?? ''}</span>
      </div>

      {job.error ? (
        <details className="mt-3 rounded-md border bg-destructive/5 p-2 text-xs">
          <summary className="cursor-pointer font-medium text-destructive">Details</summary>
          <pre className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap text-[11px]">
            {job.error}
          </pre>
        </details>
      ) : null}

      <div className="mt-3 flex items-center gap-2">
        {!isTerminal ? (
          <Button variant="outline" size="sm" onClick={cancel}>
            Cancel
          </Button>
        ) : null}
        {wsLink ? (
          <Link to={wsLink.to} params={wsLink.params} className={buttonClass('default', 'sm')}>
            <CheckCircle2 className="size-3.5" />
            Open
          </Link>
        ) : null}
        {isTerminal ? (
          <button
            onClick={onDismiss}
            className="ml-auto inline-flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="Dismiss"
          >
            <X className="size-4" />
          </button>
        ) : null}
      </div>
    </Card>
  );
}

function applyEvent(
  ev: JobEvent,
  onPatch: (patch: Partial<JobRecord>) => void,
  onTerminal: () => void,
): void {
  switch (ev.type) {
    case 'record':
      onPatch({
        state: (ev as unknown as JobRecord).state,
        progress: (ev as unknown as JobRecord).progress,
        stage: (ev as unknown as JobRecord).stage,
        message: (ev as unknown as JobRecord).message,
        slug: (ev as unknown as JobRecord).slug,
        timestamp: (ev as unknown as JobRecord).timestamp,
        workspace: (ev as unknown as JobRecord).workspace,
        error: (ev as unknown as JobRecord).error,
        finished_at: (ev as unknown as JobRecord).finished_at,
        started_at: (ev as unknown as JobRecord).started_at,
      });
      break;
    case 'state':
      onPatch({ state: ev.state });
      break;
    case 'event':
      onPatch({
        progress: ev.progress,
        stage: typeof ev.stage === 'string' ? ev.stage : null,
        message: ev.message,
      });
      break;
    case 'workspace':
      onPatch({ slug: ev.slug, timestamp: ev.timestamp, workspace: ev.workspace });
      break;
    case 'terminal':
      onPatch({
        state: ev.state,
        error: ev.error,
        finished_at: ev.finished_at,
      });
      onTerminal();
      break;
    default:
      break;
  }
}

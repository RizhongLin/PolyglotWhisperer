import { useEffect, useMemo, useRef, useState } from 'react';
import { createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import {
  ChevronRight,
  Loader2,
  Play,
  Sparkles,
  TriangleAlert,
  Upload,
} from 'lucide-react';
import { Button, buttonClass } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { JobsStrip, TERMINAL_STATES } from '@/components/studio/JobsStrip';
import { WorkerSelect } from '@/components/studio/WorkerSelect';
import { ApiError, api } from '@/api/client';
import type { FormDefaults, JobInputs, JobRecord } from '@/api/types';
import { cn } from '@/lib/cn';

export const Route = createFileRoute('/studio')({
  component: StudioPage,
});

function StudioPage() {
  const defaults = useQuery({ queryKey: ['form-defaults'], queryFn: () => api.formDefaults() });
  const [activeJobs, setActiveJobs] = useState<JobRecord[]>([]);

  // Bootstrap: re-attach to in-flight jobs after a refresh.
  useEffect(() => {
    void api.jobs().then(({ jobs }) => {
      const active = jobs.filter((j) => !TERMINAL_STATES.has(j.state));
      setActiveJobs(active);
    });
  }, []);

  const handleSubmit = (job: JobRecord) => {
    setActiveJobs((cur) => [job, ...cur.filter((j) => j.id !== job.id)]);
  };

  const handlePatch = (id: string, patch: Partial<JobRecord>) => {
    setActiveJobs((cur) => cur.map((j) => (j.id === id ? { ...j, ...patch } : j)));
  };

  // Keep terminal cards visible until the user dismisses them — they
  // hold the "Open workspace" link which is the whole reason the strip
  // sticks around after a job lands.
  const handleTerminal = (_id: string) => {};

  const dismiss = (id: string) => {
    setActiveJobs((cur) => cur.filter((j) => j.id !== id));
  };

  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Studio</h1>
        <p className="text-sm text-muted-foreground">
          Submit a new pipeline run — paste a URL or upload a file. Watch live progress; close the
          tab and come back any time.
        </p>
      </header>

      <NewJobForm defaults={defaults.data} onSubmitted={handleSubmit} />

      <JobsStrip
        jobs={activeJobs}
        onPatch={handlePatch}
        onTerminal={handleTerminal}
        onDismiss={dismiss}
      />
    </div>
  );
}

// ── New-job form ────────────────────────────────────────────────────────

interface NewJobFormProps {
  defaults: FormDefaults | undefined;
  onSubmitted: (job: JobRecord) => void;
}

function NewJobForm({ defaults, onSubmitted }: NewJobFormProps) {
  const formRef = useRef<HTMLFormElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);

  const { data: languages } = useQuery({
    queryKey: ['languages'],
    queryFn: () => api.languages(),
    staleTime: Infinity,
  });

  const langOptions = useMemo(() => {
    if (!languages) return null;
    return languages.map((li) => (
      <option key={li.code} value={li.code}>
        {li.name} ({li.code})
      </option>
    ));
  }, [languages]);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const fd = new FormData(e.currentTarget);
      const v = (k: string): string | null => {
        const x = fd.get(k);
        return typeof x === 'string' && x.trim() !== '' ? x.trim() : null;
      };
      const num = (k: string): number | null => {
        const x = v(k);
        if (x == null) return null;
        const n = Number(x);
        return Number.isFinite(n) ? n : null;
      };
      const bool = (k: string) => fd.get(k) === 'on';

      let inputPath = v('input');
      const file = fileRef.current?.files?.[0] ?? null;
      if (file) {
        const uploaded = await api.uploadFile(file);
        inputPath = uploaded.files[0]?.path ?? null;
      }
      if (!inputPath) throw new Error('Provide a URL or pick a file');
      const language = v('language');
      if (!language) throw new Error('Source language is required');

      const payload: Partial<JobInputs> & Pick<JobInputs, 'input' | 'language'> = {
        input: inputPath,
        language,
        translate: v('translate'),
        backend: (v('backend') as 'local' | 'api' | null) ?? null,
        llm_backend: (v('llm_backend') as 'local' | 'api' | null) ?? null,
        whisper_model: v('whisper_model'),
        llm_model: v('llm_model'),
        refine: bool('refine'),
        subs: bool('subs'),
        chunk_size: num('chunk_size'),
        start: v('start'),
        duration: v('duration'),
        executor: (v('executor') as 'auto' | 'worker' | 'server') ?? 'auto',
      };

      const { job_id } = await api.submitJob(payload);
      onSubmitted({
        id: job_id,
        state: 'pending',
        inputs: { ...(payload as JobInputs) },
        workspace: null,
        slug: null,
        timestamp: null,
        created_at: Date.now() / 1000,
        started_at: null,
        finished_at: null,
        error: null,
        progress: 0,
        stage: null,
        message: null,
      });
      formRef.current?.reset();
      if (fileRef.current) fileRef.current.value = '';
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="size-4 text-primary" /> New job
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form ref={formRef} onSubmit={onSubmit} className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto]">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="input">URL or local path</Label>
              <Input
                id="input"
                name="input"
                placeholder="https://… or /path/to/video.mp4"
                autoComplete="off"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="file">Or upload a file</Label>
              <label
                htmlFor="file"
                className={cn(
                  buttonClass('outline'),
                  'cursor-pointer h-9 flex items-center gap-2',
                )}
              >
                <Upload className="size-4" /> Choose file
              </label>
              <input
                id="file"
                ref={fileRef}
                name="file"
                type="file"
                className="hidden"
                onChange={(e) => {
                  const f = e.currentTarget.files?.[0];
                  if (f && formRef.current) {
                    const inp = formRef.current.elements.namedItem('input') as HTMLInputElement;
                    inp.value = f.name;
                    inp.disabled = true;
                  }
                }}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="language">Source language</Label>
              {langOptions ? (
                <Select
                  id="language"
                  name="language"
                  defaultValue={defaults?.language ?? 'fr'}
                  required
                >
                  {langOptions}
                </Select>
              ) : (
                <Input
                  id="language"
                  name="language"
                  defaultValue={defaults?.language ?? 'fr'}
                  required
                />
              )}
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="translate">Translate to (optional)</Label>
              {langOptions ? (
                <Select id="translate" name="translate" defaultValue={defaults?.translate ?? ''}>
                  <option value="">— none —</option>
                  {langOptions}
                </Select>
              ) : (
                <Input
                  id="translate"
                  name="translate"
                  defaultValue={defaults?.translate ?? ''}
                  placeholder="e.g. en, zh"
                />
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={() => setAdvanced((v) => !v)}
            className="flex items-center gap-1 self-start text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            <ChevronRight
              className={cn('size-4 transition-transform', advanced && 'rotate-90')}
            />
            Advanced
          </button>
          {advanced ? <AdvancedFields defaults={defaults} /> : null}

          {error ? (
            <p className="flex items-center gap-2 text-sm text-destructive">
              <TriangleAlert className="size-4" /> {error}
            </p>
          ) : null}

          <div className="flex justify-end">
            <Button type="submit" disabled={busy}>
              {busy ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
              <span>Start job</span>
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

function AdvancedFields({ defaults }: { defaults: FormDefaults | undefined }) {
  return (
    <div className="grid grid-cols-1 gap-4 rounded-md border bg-muted/30 p-4 md:grid-cols-2">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="backend">Whisper backend</Label>
        <Select id="backend" name="backend" defaultValue="">
          <option value="">(default: {defaults?.backend ?? 'local'})</option>
          <option value="local">local</option>
          <option value="api">api</option>
        </Select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="llm_backend">LLM backend</Label>
        <Select id="llm_backend" name="llm_backend" defaultValue="">
          <option value="">(default: {defaults?.llm_backend ?? 'local'})</option>
          <option value="local">local</option>
          <option value="api">api</option>
        </Select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="whisper_model">Whisper model</Label>
        <Input
          id="whisper_model"
          name="whisper_model"
          placeholder={defaults?.whisper_model || 'auto'}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="llm_model">LLM model</Label>
        <Input id="llm_model" name="llm_model" placeholder={defaults?.llm_model || 'auto'} />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="start">Start (ffmpeg)</Label>
        <Input id="start" name="start" placeholder="e.g. 00:01:00" />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="duration">Duration (ffmpeg)</Label>
        <Input id="duration" name="duration" placeholder="e.g. 00:05:00" />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="chunk_size">Chunk size (segments)</Label>
        <Input id="chunk_size" name="chunk_size" type="number" min={1} max={400} placeholder="auto" />
      </div>
      <div className="flex items-end gap-6">
        <label className="flex items-center gap-2 text-sm">
          <Checkbox name="refine" /> Refine
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Checkbox name="subs" /> Existing subs
        </label>
      </div>
      <WorkerSelect />
    </div>
  );
}

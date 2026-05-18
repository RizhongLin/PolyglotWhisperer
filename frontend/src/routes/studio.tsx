import { useEffect, useMemo, useRef, useState } from 'react';
import { createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import {
  Check,
  ChevronDown,
  FileVideo,
  Globe,
  Link as LinkIcon,
  Loader2,
  Play,
  Settings2,
  Sparkles,
  TriangleAlert,
  Upload,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
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

  const handleTerminal = (_id: string) => {};

  const dismiss = (id: string) => {
    setActiveJobs((cur) => cur.filter((j) => j.id !== id));
  };

  return (
    <div className="flex flex-col gap-10">
      {/* ── Hero ─────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary/10">
            <Sparkles className="size-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Studio</h1>
            <p className="text-sm text-muted-foreground">
              Submit a new pipeline run — paste a URL or upload a file
            </p>
          </div>
        </div>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Watch live progress; close the tab and come back any time. Completed
          workspaces appear in your Library.
        </p>
      </section>

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
  const [fileName, setFileName] = useState<string | null>(null);

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
      setFileName(null);
      if (fileRef.current) fileRef.current.value = '';
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="overflow-hidden">
      {/* ── Accent bar ─────────────────────────────────────────── */}
      <div className="h-1 bg-linear-to-r from-primary/60 via-primary/30 to-transparent" />

      <CardContent className="p-0">
        <form ref={formRef} onSubmit={onSubmit}>
          {/* ── Section: Source ────────────────────────────────── */}
          <div className="p-6 pb-5">
            <div className="mb-4 flex items-center gap-2">
              <FileVideo className="size-4 text-muted-foreground" />
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Source
              </h2>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto]">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="input">Video URL or local path</Label>
                <div className="relative">
                  <LinkIcon className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    id="input"
                    name="input"
                    placeholder="https://youtube.com/… or /path/to/video.mp4"
                    autoComplete="off"
                    disabled={!!fileName}
                    className="pl-9"
                  />
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <Label className="invisible md:visible">Upload</Label>
                <label
                  htmlFor="file"
                  className={cn(
                    'group flex h-9 cursor-pointer items-center gap-2 rounded-md border-2 border-dashed px-4 text-sm transition-colors',
                    fileName
                      ? 'border-primary/40 bg-primary/5 text-primary'
                      : 'border-input hover:border-primary/40 hover:bg-accent/50',
                  )}
                >
                  {fileName ? (
                    <>
                      <Check className="size-4 shrink-0" />
                      <span className="truncate">{fileName}</span>
                    </>
                  ) : (
                    <>
                      <Upload className="size-4 shrink-0 text-muted-foreground group-hover:text-foreground" />
                      <span className="text-muted-foreground group-hover:text-foreground">
                        Choose file…
                      </span>
                    </>
                  )}
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
                      setFileName(f.name);
                    }
                  }}
                />
              </div>
            </div>
          </div>

          {/* ── Divider ─────────────────────────────────────────── */}
          <div className="mx-6 border-t" />

          {/* ── Section: Languages ─────────────────────────────── */}
          <div className="p-6 py-5">
            <div className="mb-4 flex items-center gap-2">
              <Globe className="size-4 text-muted-foreground" />
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Languages
              </h2>
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
                <Label htmlFor="translate">Translate to</Label>
                {langOptions ? (
                  <Select
                    id="translate"
                    name="translate"
                    defaultValue={defaults?.translate ?? ''}
                  >
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
          </div>

          {/* ── Divider ─────────────────────────────────────────── */}
          <div className="mx-6 border-t" />

          {/* ── Section: Advanced ───────────────────────────────── */}
          <div className="p-6 pt-5">
            <button
              type="button"
              onClick={() => setAdvanced((v) => !v)}
              className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 -mx-2 text-sm font-medium transition-colors hover:bg-accent"
            >
              <Settings2 className="size-4 text-muted-foreground" />
              <span>Advanced</span>
              <ChevronDown
                className={cn(
                  'ml-auto size-4 text-muted-foreground transition-transform',
                  advanced && 'rotate-180',
                )}
              />
            </button>

            {advanced ? (
              <div className="mt-4 grid grid-cols-1 gap-4 rounded-lg border bg-muted/30 p-4 md:grid-cols-2">
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
                  <Input
                    id="llm_model"
                    name="llm_model"
                    placeholder={defaults?.llm_model || 'auto'}
                  />
                </div>

                <div className="col-span-full border-t" />

                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="start">Start (ffmpeg)</Label>
                  <Input id="start" name="start" placeholder="e.g. 00:01:00" />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="duration">Duration (ffmpeg)</Label>
                  <Input id="duration" name="duration" placeholder="e.g. 00:05:00" />
                </div>

                <div className="col-span-full border-t" />

                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="chunk_size">Chunk size (segments)</Label>
                  <Input
                    id="chunk_size"
                    name="chunk_size"
                    type="number"
                    min={1}
                    max={400}
                    placeholder="auto"
                  />
                </div>
                <div className="flex items-end gap-6">
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <Checkbox name="refine" defaultChecked={defaults?.refine} /> Refine
                  </label>
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <Checkbox name="subs" /> Existing subs
                  </label>
                </div>

                <div className="col-span-full border-t" />

                <WorkerSelect />
              </div>
            ) : null}
          </div>

          {/* ── Divider ─────────────────────────────────────────── */}
          <div className="mx-6 border-t" />

          {/* ── Section: Submit ─────────────────────────────────── */}
          <div className="flex items-center justify-between gap-4 p-6 pt-5">
            <div className="flex-1">
              {error ? (
                <p className="flex items-center gap-2 text-sm text-destructive">
                  <TriangleAlert className="size-4 shrink-0" /> {error}
                </p>
              ) : null}
            </div>
            <Button type="submit" disabled={busy} size="lg" className="gap-2">
              {busy ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              <span>Start job</span>
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

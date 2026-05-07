// Library-page entry point: new-job dialog, jobs strip, live progress,
// cancel, and reattach across browser refresh. Vanilla TypeScript — no
// framework. The server renders the workspace grid; this module only owns
// the dynamic job lifecycle UI.

import {
  ApiError,
  cancelJob,
  listJobs,
  openEventStream,
  submitJob,
  uploadFile,
  type JobRequestPayload,
} from './api.ts';
import { $, $$, escapeHtml, readBool, readNumber, readString } from './dom.ts';
import type { JobEvent, JobInputs, JobRecord, JobState } from './types.ts';

const STATE_LABEL: Record<JobState, string> = {
  pending: 'Queued',
  running: 'Running',
  cancelling: 'Cancelling…',
  cancelled: 'Cancelled',
  succeeded: 'Done',
  failed: 'Failed',
  interrupted: 'Interrupted',
};

const TERMINAL_STATES: ReadonlySet<JobState> = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
]);

interface LiveJob {
  card: HTMLElement;
  abort: (() => void) | null;
}

// ── Card rendering ───────────────────────────────────────────────────────

function fmtStage(stage: string | null | undefined): string {
  if (!stage) return 'Pending';
  return stage.charAt(0).toUpperCase() + stage.slice(1);
}

function jobCard(job: Pick<JobRecord, 'id' | 'state' | 'inputs' | 'progress' | 'stage'>): HTMLElement {
  const card = document.createElement('article');
  card.className = 'job-card';
  card.dataset.jobId = job.id;
  card.dataset.state = job.state;
  const inputs = job.inputs ?? ({} as JobInputs);
  const title = inputs.input ?? 'Job';
  const lang = inputs.language ?? '';
  const target = inputs.translate ? ` → ${inputs.translate}` : '';
  const pct = Math.round((job.progress ?? 0) * 100);
  card.innerHTML = `
    <header class="job-card-header">
      <span class="job-card-title" title="${escapeHtml(title)}">${escapeHtml(title)}</span>
      <span class="job-card-state" data-state="${escapeHtml(job.state)}">${escapeHtml(STATE_LABEL[job.state] ?? job.state)}</span>
    </header>
    <div class="job-card-meta">${escapeHtml(lang)}${escapeHtml(target)}</div>
    <div class="job-card-progress">
      <progress value="${pct}" max="100"></progress>
      <span class="job-card-stage">${escapeHtml(fmtStage(job.stage))}</span>
    </div>
    <div class="job-card-message"></div>
    <footer class="job-card-actions">
      <button class="job-cancel outline secondary" type="button">Cancel</button>
      <a class="job-open contrast" hidden href="#" target="_self">Open</a>
      <details class="job-error" hidden><summary>Details</summary><pre></pre></details>
    </footer>
  `;
  card.querySelector<HTMLButtonElement>('.job-cancel')?.addEventListener('click', () => {
    void cancelJob(job.id).catch((err: unknown) => console.warn('cancel failed', err));
  });
  return card;
}

function applyEvent(card: HTMLElement, ev: JobEvent): void {
  if (ev.type === 'state') {
    card.dataset.state = ev.state;
    const stateEl = card.querySelector<HTMLElement>('.job-card-state');
    if (stateEl) {
      stateEl.dataset.state = ev.state;
      stateEl.textContent = STATE_LABEL[ev.state] ?? ev.state;
    }
    if (ev.state === 'cancelling') {
      card.querySelector<HTMLButtonElement>('.job-cancel')?.setAttribute('disabled', 'true');
    }
    return;
  }
  if (ev.type === 'event') {
    const pct = Math.round((ev.progress || 0) * 100);
    const progress = card.querySelector<HTMLProgressElement>('progress');
    if (progress) progress.value = pct;
    const stageEl = card.querySelector<HTMLElement>('.job-card-stage');
    if (stageEl) stageEl.textContent = fmtStage(ev.stage);
    const msgEl = card.querySelector<HTMLElement>('.job-card-message');
    if (msgEl) msgEl.textContent = ev.message ?? '';
    return;
  }
  if (ev.type === 'workspace') {
    const open = card.querySelector<HTMLAnchorElement>('.job-open');
    if (open) {
      open.href = `/ws/${encodeURIComponent(ev.slug)}/${encodeURIComponent(ev.timestamp)}/`;
      open.hidden = false;
    }
    return;
  }
  if (ev.type === 'terminal') {
    card.dataset.state = ev.state;
    const stateEl = card.querySelector<HTMLElement>('.job-card-state');
    if (stateEl) {
      stateEl.dataset.state = ev.state;
      stateEl.textContent = STATE_LABEL[ev.state] ?? ev.state;
    }
    const cancelBtn = card.querySelector<HTMLButtonElement>('.job-cancel');
    if (cancelBtn) cancelBtn.hidden = true;
    if (ev.state === 'failed' || ev.state === 'interrupted') {
      const det = card.querySelector<HTMLDetailsElement>('.job-error');
      if (det) {
        det.hidden = false;
        const pre = det.querySelector('pre');
        if (pre) pre.textContent = ev.error ?? 'Job did not finish';
      }
    }
    if (ev.state === 'succeeded') scheduleLibraryRefresh();
  }
  // heartbeat / record: no-op on client.
}

// ── Live job lifecycle ───────────────────────────────────────────────────
const liveJobs = new Map<string, LiveJob>();

function ensureCard(job: JobRecord, strip: HTMLElement): HTMLElement {
  const existing = liveJobs.get(job.id);
  if (existing) return existing.card;
  const card = jobCard(job);
  strip.prepend(card);
  strip.hidden = false;
  liveJobs.set(job.id, { card, abort: null });
  const abort = openEventStream<JobEvent>(
    job.id,
    (ev) => applyEvent(card, ev),
    (err) => {
      const msgEl = card.querySelector<HTMLElement>('.job-card-message');
      if (msgEl) msgEl.textContent = `Stream interrupted: ${err.message}. Reconnecting…`;
      // Reconnect after a short delay — the server will replay the log.
      window.setTimeout(() => reattach(job.id, card), 2000);
    },
  );
  liveJobs.set(job.id, { card, abort });
  return card;
}

function reattach(jobId: string, card: HTMLElement): void {
  const slot = liveJobs.get(jobId);
  if (slot?.abort) slot.abort();
  const abort = openEventStream<JobEvent>(
    jobId,
    (ev) => applyEvent(card, ev),
    (err) => console.warn('reattach failed', err),
  );
  liveJobs.set(jobId, { card, abort });
}

// ── Library refresh after success ────────────────────────────────────────
let refreshTimer: number | null = null;
function scheduleLibraryRefresh(): void {
  if (refreshTimer !== null) return;
  refreshTimer = window.setTimeout(() => {
    refreshTimer = null;
    // Simple, robust approach: reload the page so the server-rendered
    // workspace grid picks up the new entry.
    window.location.reload();
  }, 1200);
}

// ── Form ─────────────────────────────────────────────────────────────────
function readPayload(form: HTMLFormElement): JobRequestPayload {
  const input = readString(form, 'input');
  const language = readString(form, 'language') ?? '';
  return {
    input: input ?? '',
    language,
    translate: readString(form, 'translate'),
    backend: (readString(form, 'backend') as 'local' | 'api' | null) ?? null,
    llm_backend: (readString(form, 'llm_backend') as 'local' | 'api' | null) ?? null,
    whisper_model: readString(form, 'whisper_model'),
    llm_model: readString(form, 'llm_model'),
    refine: readBool(form, 'refine'),
    subs: readBool(form, 'subs'),
    chunk_size: readNumber(form, 'chunk_size'),
    start: readString(form, 'start'),
    duration: readString(form, 'duration'),
  };
}

function wireForm(strip: HTMLElement): void {
  const dialog = $<HTMLDialogElement>('#new-job-dialog');
  const form = $<HTMLFormElement>('#new-job-form');
  const openBtn = $<HTMLButtonElement>('#new-job-open');
  const cancelBtn = $<HTMLButtonElement>('#new-job-cancel');
  const submitBtn = $<HTMLButtonElement>('#new-job-submit');
  const errEl = $<HTMLElement>('#new-job-error');
  const fileInput = $<HTMLInputElement>('#new-job-file');

  if (!dialog || !form || !openBtn || !cancelBtn || !submitBtn || !errEl) return;

  openBtn.addEventListener('click', () => {
    errEl.textContent = '';
    dialog.showModal();
  });
  cancelBtn.addEventListener('click', () => dialog.close());

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.setAttribute('aria-busy', 'true');
    submitBtn.disabled = true;
    errEl.textContent = '';
    try {
      const payload = readPayload(form);
      // If a file is selected, upload it first and use the returned path.
      if (fileInput?.files && fileInput.files.length > 0) {
        const f = fileInput.files[0];
        if (f) {
          const uploaded = await uploadFile(f);
          payload.input = uploaded.path;
        }
      }
      if (!payload.input) throw new Error('Provide a URL or pick a file');
      if (!payload.language) throw new Error('Source language is required');
      const { job_id } = await submitJob(payload);
      const stub: JobRecord = {
        id: job_id,
        state: 'pending',
        inputs: payload as JobInputs,
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
      };
      ensureCard(stub, strip);
      dialog.close();
      form.reset();
      if (fileInput) fileInput.value = '';
    } catch (err) {
      errEl.textContent = err instanceof ApiError ? err.message : (err as Error).message;
    } finally {
      submitBtn.removeAttribute('aria-busy');
      submitBtn.disabled = false;
    }
  });
}

// ── Bootstrap ────────────────────────────────────────────────────────────
async function bootstrap(): Promise<void> {
  const strip = $<HTMLElement>('#jobs-strip');
  if (!strip) return;
  wireForm(strip);
  // Re-attach to in-flight jobs so refreshing the tab does not lose progress.
  try {
    const { jobs } = await listJobs();
    for (const j of jobs) {
      if (!TERMINAL_STATES.has(j.state)) ensureCard(j, strip);
    }
  } catch (err) {
    if (!(err instanceof ApiError) || err.status !== 404) {
      console.warn('jobs listing failed', err);
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => void bootstrap());
} else {
  void bootstrap();
}

// keep the module tree-shake-safe
export {};
// silence unused-import warnings if downstream refactors trim usage
$$;

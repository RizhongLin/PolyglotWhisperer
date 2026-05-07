import type {
  FormDefaults,
  JobEvent,
  JobInputs,
  JobRecord,
  LanguageInfo,
  VocabSummary,
  WorkspaceDetail,
  WorkspaceSummary,
} from './types';

export class ApiError extends Error {
  status: number;
  payload?: unknown;

  constructor(message: string, status: number, payload?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const resp = await fetch(input, init);
  const text = await resp.text();
  let body: unknown;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!resp.ok) {
    throw new ApiError(pickError(body) ?? `HTTP ${resp.status}`, resp.status, body);
  }
  return body as T;
}

function pickError(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') return null;
  const obj = payload as Record<string, unknown>;
  if (typeof obj['error'] === 'string') return obj['error'];
  const detail = obj['detail'];
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : null))
      .filter((m): m is string => Boolean(m));
    if (msgs.length > 0) return msgs.join('; ');
  }
  return null;
}

export const api = {
  workspaces: () => request<{ workspaces: WorkspaceSummary[] }>('/api/workspaces'),

  workspace: (slug: string, ts: string) =>
    request<WorkspaceDetail>(`/api/workspaces/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}`),

  vocab: (slug: string, ts: string) =>
    request<VocabSummary>(
      `/api/workspaces/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}/vocab`,
    ),

  formDefaults: () => request<FormDefaults>('/api/config/defaults'),

  languages: () => request<LanguageInfo[]>('/api/languages'),

  jobs: () => request<{ jobs: JobRecord[] }>('/jobs'),

  job: (id: string) => request<JobRecord>(`/jobs/${encodeURIComponent(id)}`),

  cancelJob: (id: string) =>
    request<{ cancelled: boolean }>(`/jobs/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  submitJob: (payload: Partial<JobInputs> & Pick<JobInputs, 'input' | 'language'>) =>
    request<{ job_id: string }>('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  uploadFile: async (file: File) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return request<{ files: Array<{ path: string; name: string; size: number }> }>(
      '/uploads',
      { method: 'POST', body: fd },
    );
  },
};

/**
 * Open the NDJSON event stream for a job. Returns a function that aborts
 * the underlying fetch. The stream sends ``record`` / ``state`` /
 * ``event`` / ``workspace`` / ``terminal`` / ``heartbeat`` discriminated
 * payloads (see `JobEvent`). Replays from the on-disk log first then
 * tails live, so reconnect-after-refresh works without losing events.
 */
export function openJobStream(
  jobId: string,
  onEvent: (ev: JobEvent) => void,
  onError: (err: Error) => void,
): () => void {
  const ctrl = new AbortController();

  (async () => {
    try {
      const resp = await fetch(`/jobs/${encodeURIComponent(jobId)}/events`, {
        signal: ctrl.signal,
      });
      if (!resp.ok || !resp.body) {
        throw new ApiError(`Stream failed: HTTP ${resp.status}`, resp.status);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl = buf.indexOf('\n');
        while (nl !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (line) {
            try {
              onEvent(JSON.parse(line) as JobEvent);
            } catch {
              // skip malformed
            }
          }
          nl = buf.indexOf('\n');
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      onError(err as Error);
    }
  })();

  return () => ctrl.abort();
}

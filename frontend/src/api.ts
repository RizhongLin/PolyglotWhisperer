// Typed fetch helpers for the jobs API.
import type { JobInputs, JobRecord } from './types.ts';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly details?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function parseJsonOrThrow<T>(resp: Response): Promise<T> {
  const text = await resp.text();
  let payload: unknown;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { detail: text || `HTTP ${resp.status}` };
  }
  if (!resp.ok) {
    const errMsg = pickErrorMessage(payload) ?? `HTTP ${resp.status}`;
    throw new ApiError(errMsg, resp.status, payload);
  }
  return payload as T;
}

/**
 * Best-effort error-message extraction. FastAPI's ``HTTPException`` serializes
 * as ``{"detail": "..."}``; pydantic validation errors as
 * ``{"detail": [...]}``; and anything we hand-roll uses ``{"error": "..."}``.
 * Check both keys so the UI always shows the actual message rather than
 * falling back to ``HTTP 400``.
 */
function pickErrorMessage(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') return null;
  const obj = payload as Record<string, unknown>;
  if (typeof obj['error'] === 'string') return obj['error'];
  const detail = obj['detail'];
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // pydantic v2 validation: list of {loc, msg, type, ...}
    const msgs = detail
      .map((d) => {
        if (d && typeof d === 'object' && 'msg' in d) {
          return String((d as { msg: unknown }).msg);
        }
        return null;
      })
      .filter((m): m is string => Boolean(m));
    if (msgs.length > 0) return msgs.join('; ');
  }
  return null;
}

export type JobRequestPayload = Partial<JobInputs> & Pick<JobInputs, 'input' | 'language'>;

export async function submitJob(payload: JobRequestPayload): Promise<{ job_id: string }> {
  const resp = await fetch('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return parseJsonOrThrow<{ job_id: string }>(resp);
}

export async function listJobs(): Promise<{ jobs: JobRecord[] }> {
  const resp = await fetch('/jobs');
  return parseJsonOrThrow<{ jobs: JobRecord[] }>(resp);
}

export async function cancelJob(id: string): Promise<{ cancelled: boolean }> {
  const resp = await fetch(`/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' });
  return parseJsonOrThrow<{ cancelled: boolean }>(resp);
}

export async function uploadFile(file: File): Promise<{ path: string; name: string; size: number }> {
  const fd = new FormData();
  fd.append('file', file, file.name);
  const resp = await fetch('/uploads', { method: 'POST', body: fd });
  const data = await parseJsonOrThrow<{ files: Array<{ path: string; name: string; size: number }> }>(resp);
  const first = data.files[0];
  if (!first) throw new ApiError('Upload returned no files', resp.status, data);
  return first;
}

/**
 * Open the NDJSON event stream for a job and dispatch each parsed event to
 * `onEvent`. Returns a function that aborts the underlying fetch.
 *
 * The server sends:
 *   - replay-from-log lines (for reattach across browser refresh)
 *   - live events while the job runs
 *   - heartbeats every 15s during quiet stretches
 *   - a terminal event that closes the stream
 */
export function openEventStream<T>(
  jobId: string,
  onEvent: (ev: T) => void,
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
      // Stream lifetime — exits naturally on terminal event (server closes the body).
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
              onEvent(JSON.parse(line) as T);
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

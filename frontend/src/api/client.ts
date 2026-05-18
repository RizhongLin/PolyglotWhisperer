import type {
  AdminUser,
  AuthState,
  CreateFlashcardRequest,
  CreateUserRequest,
  Credential,
  CredentialCreate,
  ChangePasswordRequest,
  FlashcardResponse,
  FormDefaults,
  FsrsRating,
  JobEvent,
  JobInputs,
  JobRecord,
  LanguageInfo,
  MeResponse,
  Preferences,
  ResetPasswordRequest,
  VocabSummary,
  WorkerSummary,
  WorkspaceDetail,
  WorkspaceSummary,
} from './types';

/**
 * Stable error codes returned by the backend (mirrors `pgw/errors.py`).
 *
 * Wire format: server raises `HTTPException(detail={"code": "...", "message": "..."})`
 * — we pull the `code` out so callers can switch on it without parsing
 * free-form strings. New codes can be added freely; renames need a
 * coordinated server + client release.
 */
export const ErrCode = {
  AuthInvalidCredentials: 'auth.invalid_credentials',
  AuthNotAuthenticated: 'auth.not_authenticated',
  AuthAdminRequired: 'auth.admin_required',
  AuthInvalidEmail: 'auth.invalid_email',
  CsrfMissing: 'csrf.missing',
  CsrfMismatch: 'csrf.mismatch',
  CsrfInvalidSignature: 'csrf.invalid_signature',
  SetupAlreadyComplete: 'setup.already_complete',
  WorkerNotFound: 'worker.not_found',
  WorkerNoConnected: 'worker.no_connected_worker',
  JobNotFound: 'job.not_found',
  JobInputRejected: 'job.input_rejected',
} as const;

export type ErrCodeValue = (typeof ErrCode)[keyof typeof ErrCode];

export class ApiError extends Error {
  status: number;
  /** Backend `Err` code if present; null for legacy/free-form errors. */
  code: string | null;
  payload?: unknown;

  constructor(message: string, status: number, code: string | null, payload?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.payload = payload;
  }

  is(code: ErrCodeValue): boolean {
    return this.code === code;
  }
}

const CSRF_COOKIE = 'pgw_csrf';
const CSRF_HEADER = 'X-CSRF-Token';

function readCookie(name: string): string | null {
  // Document cookies are always plain text, no need for any parsing
  // helpers. Linear scan is fine for the handful of cookies we set.
  const prefix = `${name}=`;
  for (const part of document.cookie.split(';')) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }
  return null;
}

function isMutation(method: string | undefined): boolean {
  if (!method) return false;
  const m = method.toUpperCase();
  return m !== 'GET' && m !== 'HEAD' && m !== 'OPTIONS';
}

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (isMutation(init?.method)) {
    const csrf = readCookie(CSRF_COOKIE);
    if (csrf) headers.set(CSRF_HEADER, csrf);
  }
  const resp = await fetch(input, {
    ...init,
    headers,
    credentials: 'include',
  });
  const text = await resp.text();
  let body: unknown;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!resp.ok) {
    const { message, code } = pickError(body);
    throw new ApiError(message ?? `HTTP ${resp.status}`, resp.status, code, body);
  }
  return body as T;
}

/**
 * Extracts `{message, code}` from FastAPI error responses.
 *
 * Three shapes we accept (most → least preferred):
 *   1. `{"detail": {"code": "auth.invalid_credentials", "message": "..."}}` — current `Err.envelope()` format.
 *   2. `{"detail": "..."}` — legacy free-form string.
 *   3. `{"detail": [{"msg": "...", ...}, ...]}` — Pydantic 422 validation list.
 */
function pickError(payload: unknown): { message: string | null; code: string | null } {
  if (!payload || typeof payload !== 'object') return { message: null, code: null };
  const obj = payload as Record<string, unknown>;
  if (typeof obj['error'] === 'string') return { message: obj['error'], code: null };

  const detail = obj['detail'];
  if (typeof detail === 'string') return { message: detail, code: null };
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const env = detail as Record<string, unknown>;
    const code = typeof env['code'] === 'string' ? env['code'] : null;
    const message = typeof env['message'] === 'string' ? env['message'] : null;
    return { message, code };
  }
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) =>
        d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : null,
      )
      .filter((m): m is string => Boolean(m));
    if (msgs.length > 0) return { message: msgs.join('; '), code: null };
  }
  return { message: null, code: null };
}

export const api = {
  // ── Auth ──

  authState: () => request<AuthState>('/api/auth/state'),

  me: () => request<MeResponse>('/api/me'),

  setupAdmin: (email: string, password: string) =>
    request<{ status: string }>('/api/auth/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<null>('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    }),

  logout: () => request<null>('/api/auth/logout', { method: 'POST' }),

  // ── Workspaces ──

  workspaces: () => request<{ workspaces: WorkspaceSummary[] }>('/api/workspaces'),

  workspace: (slug: string, ts: string) =>
    request<WorkspaceDetail>(
      `/api/workspaces/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}`,
    ),

  vocab: (slug: string, ts: string) =>
    request<VocabSummary>(
      `/api/workspaces/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}/vocab`,
    ),

  markEmbedBlocked: (slug: string, ts: string) =>
    request<void>(
      `/api/workspaces/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}/embed-blocked`,
      { method: 'POST' },
    ),

  audioClipUrl: (slug: string, ts: string, startMs: number, endMs: number) =>
    `/api/workspaces/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}/audio-clip` +
    `?start=${Math.max(0, Math.floor(startMs))}&end=${Math.max(0, Math.floor(endMs))}`,

  // ── Flashcards ──

  createFlashcard: (payload: CreateFlashcardRequest) =>
    request<FlashcardResponse>('/api/flashcards', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  flashcards: (workspaceId?: number) => {
    const qs = workspaceId !== undefined ? `?workspace_id=${workspaceId}` : '';
    return request<FlashcardResponse[]>(`/api/flashcards${qs}`);
  },

  flashcardQueue: (limit = 20) =>
    request<FlashcardResponse[]>(`/api/flashcards/queue?limit=${limit}`),

  reviewFlashcard: (id: number, rating: FsrsRating, elapsedMs?: number) =>
    request<FlashcardResponse>(`/api/flashcards/${id}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating, ...(elapsedMs != null ? { elapsed_ms: elapsedMs } : {}) }),
    }),

  deleteFlashcard: (id: number) =>
    request<void>(`/api/flashcards/${id}`, { method: 'DELETE' }),

  refineFlashcard: (id: number) =>
    request<FlashcardResponse>(`/api/flashcards/${id}/refine`, { method: 'POST' }),

  formDefaults: () => request<FormDefaults>('/api/config/defaults'),

  languages: () => request<LanguageInfo[]>('/api/languages'),

  workers: () => request<WorkerSummary[]>('/api/workers'),

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
    return request<{ files: Array<{ path: string; name: string; size: number }> }>('/uploads', {
      method: 'POST',
      body: fd,
    });
  },

  // ── Credentials & preferences ──

  credentials: () => request<Credential[]>('/api/auth/credentials'),

  createCredential: (payload: CredentialCreate) =>
    request<{ ok: boolean; id: number }>('/api/auth/credentials', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  deleteCredential: (id: number) =>
    request<void>(`/api/auth/credentials/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),

  preferences: () => request<Preferences>('/api/auth/preferences'),

  updatePreferences: (payload: Preferences) =>
    request<{ ok: boolean }>('/api/auth/preferences', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  changePassword: (payload: ChangePasswordRequest) =>
    request<{ ok: boolean }>('/api/auth/password', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  // ── Admin ──

  adminUsers: () => request<AdminUser[]>('/api/admin/users'),

  adminCreateUser: (payload: CreateUserRequest) =>
    request<{ ok: boolean }>('/api/admin/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  adminResetPassword: (userId: number, payload: ResetPasswordRequest) =>
    request<{ ok: boolean }>(`/api/admin/users/${encodeURIComponent(userId)}/password`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  adminDeleteUser: (userId: number) =>
    request<void>(`/api/admin/users/${encodeURIComponent(userId)}`, {
      method: 'DELETE',
    }),
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
        credentials: 'include',
      });
      if (!resp.ok || !resp.body) {
        throw new ApiError(`Stream failed: HTTP ${resp.status}`, resp.status, null);
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

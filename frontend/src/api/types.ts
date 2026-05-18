// Backend wire shapes. Mirror server/app.py and server/jobs.py.

// ── Auth ──

export interface AuthState {
  has_admin: boolean;
  authenticated: boolean;
}

export interface MeResponse {
  id: number;
  email: string;
  is_admin: boolean;
}

export interface WorkspaceSummary {
  slug: string;
  timestamp: string;
  title: string;
  language: string | null;
  target_language: string | null;
  lang_pairs: Array<[string, string] | string[]>;
  duration: number | null;
  created_at: string | null;
  upload_date: string | null;
  uploader: string | null;
  thumbnail: string | null;
  difficulty: string | null;
  has_video: boolean;
}

export interface SubtitleTrack {
  file: string;
  label: string;
  lang: string;
}

export interface WorkspaceFile {
  name: string;
  size: number;
  suffix: string;
}

export interface WorkspaceMeta {
  title?: string;
  language?: string;
  target_language?: string;
  source_url?: string;
  source_duration?: number;
  uploader?: string;
  upload_date?: string;
  thumbnail?: string;
  description?: string;
  whisper_model?: string;
  llm_model?: string;
  refine?: boolean;
  start?: string;
  duration?: string;
  created_at?: string;
}

export interface EmbedTarget {
  provider: 'youtube' | 'vimeo';
  url: string;
  video_id: string;
}

export interface WorkspaceDetail {
  slug: string;
  timestamp: string;
  /** DB id, populated lazily by the detail endpoint. May be null if the
   * upsert fails (e.g. bootstrap-mode pre-admin). Flashcard creation
   * needs this. */
  id: number | null;
  metadata: WorkspaceMeta;
  tracks: SubtitleTrack[];
  files: WorkspaceFile[];
  video: string | null;
  /** Direct CDN stream URL resolved by yt-dlp. Browser can play this
   * natively via <video> — no local file or embed needed. */
  video_url: string | null;
  embed: EmbedTarget | null;
  siblings: Array<{ slug: string; timestamp: string }>;
}

export type FsrsRating = 1 | 2 | 3 | 4;

export type RefineStatus = 'pending' | 'done' | 'failed' | 'skipped';

export interface FlashcardResponse {
  id: number;
  workspace_id: number;
  workspace_slug: string;
  workspace_timestamp: string;
  vocab_entry_id: number | null;
  front: string;
  back: string;
  language: string;
  audio_start_ms: number | null;
  audio_end_ms: number | null;
  /** LLM-refined fields. Render the original front/back if any of
   * these are missing or refine_status !== 'done'. */
  lemma: string | null;
  pos: string | null;
  definition: string | null;
  example_source: string | null;
  example_target: string | null;
  mnemonic: string | null;
  refine_status: RefineStatus;
  refine_model: string | null;
  refined_at: string | null;
  fsrs_due: string;
  fsrs_stability: number;
  fsrs_difficulty: number;
  created_at: string;
  updated_at: string;
}

export interface CreateFlashcardRequest {
  workspace_id: number;
  front: string;
  back: string;
  language: string;
  audio_start_ms?: number;
  audio_end_ms?: number;
  vocab_entry_id?: number;
}

export interface VocabWord {
  word: string;
  lemma?: string;
  pos?: string;
  context?: string;
  translation?: string;
  zipf?: number;
  difficulty?: string;
  count?: number;
}

export interface VocabSummary {
  total_words?: number;
  unique_lemmas?: number;
  estimated_difficulty?: string;
  difficulty_distribution?: Record<string, number>;
  top_rare_words?: VocabWord[];
}

export interface FormDefaults {
  language: string;
  translate: string;
  backend: string;
  llm_backend: string;
  whisper_model: string;
  llm_model: string;
  refine: boolean;
}

export interface LanguageInfo {
  code: string;
  name: string;
  has_spacy: boolean;
  has_alignment: boolean;
}

export interface WorkerSummary {
  id: number;
  name: string;
  connected: boolean;
  created_at: string;
  revoked_at: string | null;
  last_seen_at: string | null;
}

// ── Jobs ──

export type JobState =
  | 'pending'
  | 'running'
  | 'cancelling'
  | 'cancelled'
  | 'succeeded'
  | 'failed'
  | 'interrupted';

export interface JobInputs {
  input: string;
  language: string;
  translate: string | null;
  backend: 'local' | 'api' | null;
  llm_backend: 'local' | 'api' | null;
  whisper_model: string | null;
  llm_model: string | null;
  refine: boolean;
  subs: boolean;
  chunk_size: number | null;
  start: string | null;
  duration: string | null;
  executor: 'auto' | 'worker' | 'server';
}

export interface JobRecord {
  id: string;
  state: JobState;
  inputs: JobInputs;
  workspace: string | null;
  slug: string | null;
  timestamp: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  progress: number;
  stage: string | null;
  message: string | null;
}

export type JobStage =
  | 'download'
  | 'audio'
  | 'transcribe'
  | 'translate'
  | 'vocab'
  | 'save';

export type JobEvent =
  | { type: 'record'; [k: string]: unknown }
  | { type: 'state'; state: JobState; ts: number; started_at?: number }
  | {
      type: 'event';
      stage: JobStage | string;
      progress: number;
      message: string;
      data: Record<string, unknown> | null;
      ts: number;
    }
  | {
      type: 'workspace';
      workspace: string;
      slug: string;
      timestamp: string;
      ts: number;
    }
  | {
      type: 'terminal';
      state: JobState;
      error: string | null;
      finished_at: number | null;
      ts: number;
    }
  | { type: 'heartbeat'; ts: number };

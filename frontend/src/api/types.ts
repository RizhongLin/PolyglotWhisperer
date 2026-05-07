// Backend wire shapes. Mirror server/app.py and server/jobs.py.

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

export interface WorkspaceDetail {
  slug: string;
  timestamp: string;
  metadata: WorkspaceMeta;
  tracks: SubtitleTrack[];
  files: WorkspaceFile[];
  video: string | null;
  siblings: Array<{ slug: string; timestamp: string }>;
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
}

export interface LanguageInfo {
  code: string;
  name: string;
  has_spacy: boolean;
  has_alignment: boolean;
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

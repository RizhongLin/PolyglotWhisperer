// Wire-format types shared with the FastAPI backend.
// Keep these in sync with src/pgw/server/jobs.py (JobRequest, JobRecord)
// and the events emitted by JobManager._fanout_log_locked.

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

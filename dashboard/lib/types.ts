export interface Episode {
  episode_index: number;
  frames: number;
  language_instruction: string;
  action_type: string;
  quality: number;
  duration_s?: number;
}

export interface Session {
  session_id: string;
  scenario: string;
  chunks: number;
  episodes: number;
  duration_s: number;
  status: string;
  created_at: string;
  location?: string;
  notes?: string;
}

export interface PipelineStage {
  name: string;
  status: 'done' | 'running' | 'error' | 'pending' | 'skipped';
  elapsed_s?: number;
  error?: string;
}

export interface SessionStatus {
  session_id: string;
  pipeline_stages: PipelineStage[];
  last_run?: string;
  overall_status: string;
}

export interface SessionDetail extends Session {
  episodes_list: Episode[];
  gps_points?: Array<{ lat: number; lon: number; timestamp: string }>;
  video_files?: string[];
  audio_files?: string[];
}

export interface JobState {
  job_id: string;
  session_id: string;
  stage: string;
  progress: number;
  status: 'queued' | 'running' | 'done' | 'error';
  started_at?: string;
  finished_at?: string;
  duration_s?: number;
  error?: string;
}

export interface AutoLabel {
  session_id: string;
  transcription?: string;
  suggested_labels: Array<{
    action_type: string;
    confidence: number;
    instruction: string;
  }>;
}

export interface LabelingStats {
  project_id: string;
  total_annotated: number;
  pending: number;
  by_action_type: Record<string, number>;
  sessions: Array<{
    session_id: string;
    tasks_pushed: number;
    annotations_done: number;
    completion_pct: number;
  }>;
}

export interface TrainingJob {
  job_id: string;
  status: 'idle' | 'running' | 'done' | 'error';
  epoch: number;
  total_epochs: number;
  loss: number;
  eta_s?: number;
  started_at?: string;
  config?: TrainingConfig;
}

export interface Checkpoint {
  name: string;
  epoch: number;
  loss: number;
  created_at: string;
  path?: string;
  hf_pushed?: boolean;
}

export interface TrainingConfig {
  dataset_dir: string;
  epochs: number;
  lr: number;
  batch_size: number;
  output_dir?: string;
}

export interface PipelineRunRequest {
  session_id: string;
  scenario: string;
  upload_hf?: boolean;
  label?: boolean;
}

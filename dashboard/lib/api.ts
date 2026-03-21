import type {
  Session,
  SessionDetail,
  SessionStatus,
  JobState,
  AutoLabel,
  LabelingStats,
  TrainingJob,
  Checkpoint,
  TrainingConfig,
  PipelineRunRequest,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// Sessions
export async function getSessions(): Promise<Session[]> {
  return apiFetch<Session[]>('/sessions');
}

export async function getSession(id: string): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`/sessions/${id}`);
}

export async function getSessionStatus(id: string): Promise<SessionStatus> {
  return apiFetch<SessionStatus>(`/api/ingest/session/${id}/status`);
}

// Pipeline
export async function runPipeline(
  sessionId: string,
  scenario: string,
  opts?: { upload_hf?: boolean; label?: boolean }
): Promise<{ job_id: string }> {
  const body: PipelineRunRequest = {
    session_id: sessionId,
    scenario,
    upload_hf: opts?.upload_hf ?? false,
    label: opts?.label ?? false,
  };
  return apiFetch<{ job_id: string }>(`/pipeline/preprocess/${sessionId}`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function getPipelineStatus(): Promise<{
  active: JobState[];
  completed: JobState[];
}> {
  return apiFetch<{ active: JobState[]; completed: JobState[] }>('/pipeline/status');
}

// Labeling
export async function getAutoLabels(sessionId: string): Promise<AutoLabel> {
  return apiFetch<AutoLabel>(`/labeling/auto/${sessionId}`);
}

export async function pushToLabelStudio(
  sessionId: string,
  projectTitle: string
): Promise<{ task_ids: number[]; project_id: number }> {
  return apiFetch<{ task_ids: number[]; project_id: number }>(
    `/labeling/push/${sessionId}`,
    {
      method: 'POST',
      body: JSON.stringify({ project_title: projectTitle }),
    }
  );
}

export async function getLabelingStats(
  projectId: string
): Promise<LabelingStats> {
  return apiFetch<LabelingStats>(`/labeling/stats/${projectId}`);
}

// Training
export async function getTrainingJobs(): Promise<{
  active: TrainingJob | null;
  jobs: TrainingJob[];
}> {
  return apiFetch<{ active: TrainingJob | null; jobs: TrainingJob[] }>(
    '/training/status'
  );
}

export async function getCheckpoints(): Promise<Checkpoint[]> {
  return apiFetch<Checkpoint[]>('/training/checkpoints');
}

export async function startTraining(
  config: TrainingConfig
): Promise<{ job_id: string }> {
  return apiFetch<{ job_id: string }>('/training/start', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function exportToHF(
  checkpointName: string
): Promise<{ hf_url: string }> {
  return apiFetch<{ hf_url: string }>(`/training/export/${checkpointName}`, {
    method: 'POST',
  });
}

// SWR fetcher
export const fetcher = async (url: string) => {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
};

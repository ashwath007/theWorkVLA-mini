'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { clsx } from 'clsx';
import { format } from 'date-fns';
import { Play, RefreshCw, AlertTriangle } from 'lucide-react';
import { API_BASE } from '@/lib/config';
import type { JobState } from '@/lib/types';

function ErrorBanner() {
  return (
    <div className="bg-red-900/20 border border-red-700 rounded-xl p-4 flex items-start gap-3">
      <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
      <div>
        <div className="text-red-400 font-medium">API offline</div>
        <div className="text-red-300/70 text-sm">
          Start server with:{' '}
          <code className="font-mono bg-red-900/40 px-1 rounded">
            uvicorn src.api.main:app --reload
          </code>
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-500"
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-9 text-right">{value}%</span>
    </div>
  );
}

const statusBadge: Record<string, string> = {
  queued: 'bg-gray-800 text-gray-400',
  running: 'bg-yellow-900/50 text-yellow-400 animate-pulse',
  done: 'bg-green-900/50 text-green-400',
  error: 'bg-red-900/50 text-red-400',
};

function JobRow({ job, idx }: { job: JobState; idx: number }) {
  const duration =
    job.duration_s != null
      ? `${job.duration_s.toFixed(0)}s`
      : job.started_at
      ? `${Math.floor((Date.now() - new Date(job.started_at).getTime()) / 1000)}s`
      : '—';

  return (
    <tr
      className={clsx(
        'border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors',
        idx % 2 === 0 ? 'bg-transparent' : 'bg-gray-900/40'
      )}
    >
      <td className="px-4 py-3 font-mono text-xs text-gray-400">{job.job_id}</td>
      <td className="px-4 py-3 font-mono text-indigo-400 text-sm">{job.session_id}</td>
      <td className="px-4 py-3 text-sm text-gray-300 capitalize">{job.stage}</td>
      <td className="px-4 py-3 w-40">
        <ProgressBar value={job.progress ?? 0} />
      </td>
      <td className="px-4 py-3">
        <span className={clsx('text-xs px-2 py-0.5 rounded-full capitalize font-medium', statusBadge[job.status] ?? 'bg-gray-800 text-gray-400')}>
          {job.status}
        </span>
      </td>
      <td className="px-4 py-3 text-xs text-gray-400">{duration}</td>
    </tr>
  );
}

export default function PipelinePage() {
  const [runForm, setRunForm] = useState({
    session_id: '',
    scenario: 'delivery',
    upload_hf: false,
    label: false,
  });
  const [runResult, setRunResult] = useState<string>('');

  const { data, error, isLoading, mutate } = useSWR<{
    active: JobState[];
    completed: JobState[];
  }>(
    `${API_BASE}/pipeline/status`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    {
      refreshInterval: (data) => {
        if (data?.active && data.active.length > 0) return 3000;
        return 10000;
      },
    }
  );

  const active = data?.active ?? [];
  const completed = (data?.completed ?? []).slice(0, 20);

  async function handleRun() {
    if (!runForm.session_id.trim()) {
      setRunResult('Enter a session ID');
      return;
    }
    try {
      const r = await fetch(
        `${API_BASE}/pipeline/preprocess/${runForm.session_id}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(runForm),
        }
      );
      if (r.ok) {
        const d = await r.json();
        setRunResult(`Job queued: ${d.job_id ?? 'ok'}`);
        mutate();
      } else {
        setRunResult(`Error ${r.status}: ${await r.text()}`);
      }
    } catch {
      setRunResult('API offline');
    }
    setTimeout(() => setRunResult(''), 8000);
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Pipeline</h1>
          <p className="text-sm text-gray-400 mt-1">
            Monitor preprocessing jobs and trigger new runs
          </p>
        </div>
        <button
          onClick={() => mutate()}
          className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-100 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {error && <ErrorBanner />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Jobs table - left side */}
        <div className="lg:col-span-2 space-y-6">
          {/* Active jobs */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-200">Active Jobs</h2>
              {active.length > 0 && (
                <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
              )}
            </div>
            {isLoading ? (
              <div className="p-6 space-y-2">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-10 bg-gray-800 rounded animate-pulse" />
                ))}
              </div>
            ) : active.length === 0 ? (
              <div className="p-8 text-center text-gray-500 text-sm">
                No active jobs
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase tracking-wider">
                    <th className="px-4 py-3 text-left">Job ID</th>
                    <th className="px-4 py-3 text-left">Session</th>
                    <th className="px-4 py-3 text-left">Stage</th>
                    <th className="px-4 py-3 text-left">Progress</th>
                    <th className="px-4 py-3 text-left">Status</th>
                    <th className="px-4 py-3 text-left">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {active.map((job, i) => (
                    <JobRow key={job.job_id} job={job} idx={i} />
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Completed jobs */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-800">
              <h2 className="text-sm font-semibold text-gray-200">
                Completed Jobs{' '}
                <span className="text-gray-500 font-normal">(last 20)</span>
              </h2>
            </div>
            {!isLoading && completed.length === 0 ? (
              <div className="p-8 text-center text-gray-500 text-sm">
                No completed jobs yet
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase tracking-wider">
                    <th className="px-4 py-3 text-left">Job ID</th>
                    <th className="px-4 py-3 text-left">Session</th>
                    <th className="px-4 py-3 text-left">Stage</th>
                    <th className="px-4 py-3 text-left">Progress</th>
                    <th className="px-4 py-3 text-left">Status</th>
                    <th className="px-4 py-3 text-left">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoading
                    ? [...Array(5)].map((_, i) => (
                        <tr key={i}>
                          <td colSpan={6} className="px-4 py-3">
                            <div className="h-4 bg-gray-800 rounded animate-pulse" />
                          </td>
                        </tr>
                      ))
                    : completed.map((job, i) => (
                        <JobRow key={job.job_id} job={job} idx={i} />
                      ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Run pipeline form - right side */}
        <div>
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-sm font-semibold text-gray-200 mb-4">
              Run Pipeline
            </h2>
            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-400 block mb-1.5">
                  Session ID *
                </label>
                <input
                  type="text"
                  value={runForm.session_id}
                  onChange={(e) =>
                    setRunForm((f) => ({ ...f, session_id: e.target.value }))
                  }
                  placeholder="e.g. session_20240115"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1.5">
                  Scenario
                </label>
                <select
                  value={runForm.scenario}
                  onChange={(e) =>
                    setRunForm((f) => ({ ...f, scenario: e.target.value }))
                  }
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
                >
                  <option value="delivery">Delivery</option>
                  <option value="driving">Driving</option>
                  <option value="warehouse">Warehouse</option>
                  <option value="kitchen">Kitchen</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-xs text-gray-400 block">Options</label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={runForm.upload_hf}
                    onChange={(e) =>
                      setRunForm((f) => ({ ...f, upload_hf: e.target.checked }))
                    }
                    className="w-4 h-4 rounded bg-gray-800 border-gray-600 accent-indigo-500"
                  />
                  <span className="text-sm text-gray-300">Upload to HuggingFace</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={runForm.label}
                    onChange={(e) =>
                      setRunForm((f) => ({ ...f, label: e.target.checked }))
                    }
                    className="w-4 h-4 rounded bg-gray-800 border-gray-600 accent-indigo-500"
                  />
                  <span className="text-sm text-gray-300">Auto-label & push to LabelStudio</span>
                </label>
              </div>
              {runResult && (
                <div className="text-xs text-gray-300 bg-gray-800 rounded-lg px-3 py-2 font-mono">
                  {runResult}
                </div>
              )}
              <button
                onClick={handleRun}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors"
              >
                <Play className="w-4 h-4" />
                Run Pipeline
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

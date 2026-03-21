'use client';

import { use, useState } from 'react';
import Link from 'next/link';
import useSWR from 'swr';
import { clsx } from 'clsx';
import { format } from 'date-fns';
import { ArrowLeft, Play, Tag, AlertTriangle, ChevronRight } from 'lucide-react';
import { StatCard } from '@/components/StatCard';
import { StageProgress } from '@/components/StageProgress';
import { API_BASE } from '@/lib/config';
import type { SessionDetail, SessionStatus } from '@/lib/types';

function Skeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 w-64 bg-gray-800 rounded" />
      <div className="grid grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-28 bg-gray-800 rounded-xl" />
        ))}
      </div>
      <div className="h-48 bg-gray-800 rounded-xl" />
      <div className="h-64 bg-gray-800 rounded-xl" />
    </div>
  );
}

function ErrorState({ id }: { id: string }) {
  return (
    <div className="bg-red-900/20 border border-red-700 rounded-xl p-6 flex items-start gap-3">
      <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
      <div>
        <div className="text-red-400 font-medium">Could not load session {id}</div>
        <div className="text-red-300/70 text-sm mt-1">
          API offline — start server with{' '}
          <code className="font-mono bg-red-900/40 px-1 rounded">
            uvicorn src.api.main:app --reload
          </code>
        </div>
      </div>
    </div>
  );
}

const qualityColor = (q: number) => {
  if (q >= 0.8) return 'text-green-400';
  if (q >= 0.5) return 'text-yellow-400';
  return 'text-red-400';
};

export default function SessionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [runStatus, setRunStatus] = useState<string>('');
  const [pushStatus, setPushStatus] = useState<string>('');

  const { data: session, error: sessionError, isLoading } = useSWR<SessionDetail>(
    `${API_BASE}/sessions/${id}`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    { refreshInterval: 5000 }
  );

  const { data: status } = useSWR<SessionStatus>(
    `${API_BASE}/api/ingest/session/${id}/status`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    { refreshInterval: 3000 }
  );

  async function handleRunPipeline() {
    setRunStatus('running');
    try {
      const r = await fetch(`${API_BASE}/pipeline/preprocess/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: id, scenario: session?.scenario ?? 'delivery' }),
      });
      if (r.ok) {
        const d = await r.json();
        setRunStatus(`Job started: ${d.job_id ?? 'ok'}`);
      } else {
        setRunStatus(`Error: ${r.status}`);
      }
    } catch {
      setRunStatus('API offline');
    }
    setTimeout(() => setRunStatus(''), 5000);
  }

  async function handlePushLabels() {
    setPushStatus('pushing...');
    try {
      const r = await fetch(`${API_BASE}/labeling/push/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_title: `VLA ${id}` }),
      });
      if (r.ok) {
        const d = await r.json();
        setPushStatus(`Pushed ${d.task_ids?.length ?? 0} tasks`);
      } else {
        setPushStatus(`Error: ${r.status}`);
      }
    } catch {
      setPushStatus('API offline');
    }
    setTimeout(() => setPushStatus(''), 5000);
  }

  if (isLoading) return <Skeleton />;
  if (sessionError) return <ErrorState id={id} />;

  const durationH = session ? (session.duration_s / 3600).toFixed(2) : '—';

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <Link href="/sessions" className="hover:text-gray-100 flex items-center gap-1">
          <ArrowLeft className="w-3 h-3" />
          Sessions
        </Link>
        <ChevronRight className="w-3 h-3" />
        <span className="font-mono text-gray-200">{id}</span>
      </div>

      {/* Header with actions */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100 font-mono">{id}</h1>
          <div className="flex items-center gap-3 mt-1">
            <span className="text-sm text-gray-400 capitalize">
              {session?.scenario ?? '—'}
            </span>
            {session?.status && (
              <span
                className={clsx(
                  'text-xs px-2 py-0.5 rounded-full font-medium capitalize',
                  session.status === 'done' && 'bg-green-900/50 text-green-400',
                  session.status === 'running' && 'bg-yellow-900/50 text-yellow-400',
                  session.status === 'error' && 'bg-red-900/50 text-red-400',
                  (session.status === 'pending' || session.status === 'new') &&
                    'bg-gray-800 text-gray-400'
                )}
              >
                {session.status}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {runStatus && (
            <span className="text-xs text-gray-400 bg-gray-800 px-3 py-1.5 rounded-lg">
              {runStatus}
            </span>
          )}
          {pushStatus && (
            <span className="text-xs text-gray-400 bg-gray-800 px-3 py-1.5 rounded-lg">
              {pushStatus}
            </span>
          )}
          <button
            onClick={handlePushLabels}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600/20 hover:bg-purple-600 text-purple-400 hover:text-white rounded-lg text-sm font-medium transition-colors border border-purple-700"
          >
            <Tag className="w-4 h-4" />
            Push to LabelStudio
          </button>
          <button
            onClick={handleRunPipeline}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Play className="w-4 h-4" />
            Run Pipeline
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard title="Chunks" value={session?.chunks ?? '—'} subtitle="data chunks" />
        <StatCard
          title="Episodes"
          value={session?.episodes ?? '—'}
          subtitle="recorded"
          accent="indigo"
        />
        <StatCard
          title="Duration"
          value={`${durationH}h`}
          subtitle="total recording"
          accent="green"
        />
        <StatCard
          title="Status"
          value={session?.status ?? '—'}
          subtitle="pipeline state"
          accent={
            session?.status === 'done'
              ? 'green'
              : session?.status === 'error'
              ? 'red'
              : session?.status === 'running'
              ? 'yellow'
              : 'default'
          }
        />
      </div>

      {/* Pipeline stages */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-200 mb-4">
          Pipeline Stages
        </h2>
        <StageProgress stages={status?.pipeline_stages ?? []} />
        {status?.last_run && (
          <div className="mt-3 text-xs text-gray-500">
            Last run:{' '}
            {format(new Date(status.last_run), 'MMM d, yyyy HH:mm:ss')}
          </div>
        )}
      </div>

      {/* GPS placeholder */}
      {session?.gps_points && session.gps_points.length > 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-200 mb-4">
            GPS Route ({session.gps_points.length} points)
          </h2>
          <div className="overflow-auto max-h-48">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-400">
                  <th className="text-left py-1">Timestamp</th>
                  <th className="text-right py-1">Lat</th>
                  <th className="text-right py-1">Lon</th>
                </tr>
              </thead>
              <tbody>
                {session.gps_points.slice(0, 20).map((pt, i) => (
                  <tr key={i} className="border-t border-gray-800/50">
                    <td className="py-1 text-gray-400 font-mono">{pt.timestamp}</td>
                    <td className="py-1 text-right text-gray-300 font-mono">{pt.lat.toFixed(6)}</td>
                    <td className="py-1 text-right text-gray-300 font-mono">{pt.lon.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-200 mb-3">
            GPS Route Visualization
          </h2>
          <div className="h-32 bg-gray-800 rounded-lg flex items-center justify-center text-gray-500 text-sm">
            GPS route data will appear here after pipeline runs gps stage
          </div>
        </div>
      )}

      {/* Episodes table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800">
          <h2 className="text-sm font-semibold text-gray-200">
            Recent Episodes{' '}
            {session?.episodes_list && (
              <span className="text-gray-500 font-normal">
                (showing first 10)
              </span>
            )}
          </h2>
        </div>
        {!session?.episodes_list || session.episodes_list.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">
            No episodes yet. Run the pipeline to generate episodes.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase tracking-wider">
                <th className="px-4 py-3 text-right">#</th>
                <th className="px-4 py-3 text-right">Frames</th>
                <th className="px-4 py-3 text-left">Language Instruction</th>
                <th className="px-4 py-3 text-left">Action Type</th>
                <th className="px-4 py-3 text-right">Quality</th>
              </tr>
            </thead>
            <tbody>
              {session.episodes_list.slice(0, 10).map((ep, idx) => (
                <tr
                  key={ep.episode_index}
                  className={clsx(
                    'border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors',
                    idx % 2 === 0 ? 'bg-transparent' : 'bg-gray-900/50'
                  )}
                >
                  <td className="px-4 py-3 text-right text-gray-400 font-mono">
                    {ep.episode_index}
                  </td>
                  <td className="px-4 py-3 text-right text-gray-300">
                    {ep.frames}
                  </td>
                  <td className="px-4 py-3 text-gray-300 max-w-xs truncate">
                    {ep.language_instruction || '—'}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-300 capitalize">
                      {ep.action_type || '—'}
                    </span>
                  </td>
                  <td className={clsx('px-4 py-3 text-right font-mono font-medium', qualityColor(ep.quality))}>
                    {(ep.quality * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

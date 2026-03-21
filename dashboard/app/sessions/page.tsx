'use client';

import { useState } from 'react';
import Link from 'next/link';
import useSWR from 'swr';
import { format } from 'date-fns';
import { clsx } from 'clsx';
import { Upload, Play, RefreshCw, AlertTriangle } from 'lucide-react';
import { StatCard } from '@/components/StatCard';
import { API_BASE } from '@/lib/config';
import type { Session } from '@/lib/types';

const scenarioBadge: Record<string, string> = {
  delivery: 'bg-blue-900/50 text-blue-300 border border-blue-700',
  driving: 'bg-green-900/50 text-green-300 border border-green-700',
  warehouse: 'bg-orange-900/50 text-orange-300 border border-orange-700',
  kitchen: 'bg-purple-900/50 text-purple-300 border border-purple-700',
};

const statusDot: Record<string, string> = {
  done: 'bg-green-500',
  running: 'bg-yellow-500 animate-pulse',
  error: 'bg-red-500',
  pending: 'bg-gray-500',
  new: 'bg-gray-500',
};

function Skeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="grid grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-28 bg-gray-800 rounded-xl" />
        ))}
      </div>
      <div className="h-64 bg-gray-800 rounded-xl" />
    </div>
  );
}

function ErrorState() {
  return (
    <div className="bg-red-900/20 border border-red-700 rounded-xl p-6 flex items-start gap-3">
      <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
      <div>
        <div className="text-red-400 font-medium">API offline</div>
        <div className="text-red-300/70 text-sm mt-1">
          Start server with:{' '}
          <code className="font-mono bg-red-900/40 px-1 rounded">
            uvicorn src.api.main:app --reload
          </code>
        </div>
      </div>
    </div>
  );
}

export default function SessionsPage() {
  const [uploadOpen, setUploadOpen] = useState(false);
  const { data: sessions, error, isLoading, mutate } = useSWR<Session[]>(
    `${API_BASE}/sessions`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    { refreshInterval: 5000 }
  );

  const totalSessions = sessions?.length ?? 0;
  const totalEpisodes = sessions?.reduce((s, x) => s + (x.episodes ?? 0), 0) ?? 0;
  const totalDurationH = (
    (sessions?.reduce((s, x) => s + (x.duration_s ?? 0), 0) ?? 0) / 3600
  ).toFixed(1);
  const validEpisodes =
    totalEpisodes > 0
      ? Math.round(
          ((sessions?.filter((s) => s.status === 'done').reduce(
            (a, x) => a + (x.episodes ?? 0),
            0
          ) ?? 0) /
            totalEpisodes) *
            100
        )
      : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Sessions</h1>
          <p className="text-sm text-gray-400 mt-1">
            Manage and monitor data collection sessions
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => mutate()}
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-100 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setUploadOpen(true)}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            <Upload className="w-4 h-4" />
            Upload Session
          </button>
        </div>
      </div>

      {/* Error */}
      {error && <ErrorState />}

      {/* Loading */}
      {isLoading && <Skeleton />}

      {/* Stats */}
      {!isLoading && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard title="Total Sessions" value={totalSessions} subtitle="all time" />
            <StatCard title="Total Episodes" value={totalEpisodes} subtitle="across sessions" />
            <StatCard
              title="Total Duration"
              value={`${totalDurationH}h`}
              subtitle="recorded hours"
              accent="indigo"
            />
            <StatCard
              title="Valid Episodes %"
              value={`${validEpisodes}%`}
              subtitle="from done sessions"
              accent={validEpisodes >= 80 ? 'green' : validEpisodes >= 50 ? 'yellow' : 'red'}
            />
          </div>

          {/* Table */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-800">
              <h2 className="text-sm font-semibold text-gray-200">All Sessions</h2>
            </div>
            {(!sessions || sessions.length === 0) && !error ? (
              <div className="p-12 text-center text-gray-500">
                No sessions found. Upload your first session to get started.
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase tracking-wider">
                    <th className="px-4 py-3 text-left">Session ID</th>
                    <th className="px-4 py-3 text-left">Scenario</th>
                    <th className="px-4 py-3 text-right">Chunks</th>
                    <th className="px-4 py-3 text-right">Episodes</th>
                    <th className="px-4 py-3 text-left">Status</th>
                    <th className="px-4 py-3 text-left">Created At</th>
                    <th className="px-4 py-3 text-left">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {(sessions ?? []).map((s, idx) => (
                    <tr
                      key={s.session_id}
                      className={clsx(
                        'border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors',
                        idx % 2 === 0 ? 'bg-transparent' : 'bg-gray-900/50'
                      )}
                    >
                      <td className="px-4 py-3">
                        <Link
                          href={`/sessions/${s.session_id}`}
                          className="font-mono text-indigo-400 hover:text-indigo-300 transition-colors"
                        >
                          {s.session_id}
                        </Link>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={clsx(
                            'text-xs px-2 py-0.5 rounded capitalize font-medium',
                            scenarioBadge[s.scenario] ??
                              'bg-gray-800 text-gray-300 border border-gray-700'
                          )}
                        >
                          {s.scenario}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">
                        {s.chunks}
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">
                        {s.episodes}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div
                            className={clsx(
                              'w-2 h-2 rounded-full',
                              statusDot[s.status] ?? 'bg-gray-500'
                            )}
                          />
                          <span className="text-gray-300 capitalize">{s.status}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-gray-400">
                        {s.created_at
                          ? format(new Date(s.created_at), 'MMM d, yyyy HH:mm')
                          : '—'}
                      </td>
                      <td className="px-4 py-3">
                        <Link
                          href={`/sessions/${s.session_id}`}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600/20 hover:bg-indigo-600 text-indigo-400 hover:text-white rounded-lg text-xs font-medium transition-colors"
                        >
                          <Play className="w-3 h-3" />
                          Run Pipeline
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}

      {/* Upload Modal */}
      {uploadOpen && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">
              Upload Session
            </h3>
            <p className="text-sm text-gray-400 mb-6">
              Upload session files via the API endpoint POST /api/ingest/upload
            </p>
            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Session ID
                </label>
                <input
                  type="text"
                  placeholder="e.g. session_20240115_delivery"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Scenario
                </label>
                <select className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-indigo-500">
                  <option value="delivery">Delivery</option>
                  <option value="driving">Driving</option>
                  <option value="warehouse">Warehouse</option>
                  <option value="kitchen">Kitchen</option>
                </select>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setUploadOpen(false)}
                className="flex-1 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => setUploadOpen(false)}
                className="flex-1 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

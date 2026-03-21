'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { clsx } from 'clsx';
import { AlertTriangle, Tag, Loader } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { StatCard } from '@/components/StatCard';
import { API_BASE } from '@/lib/config';
import type { LabelingStats, AutoLabel } from '@/lib/types';

const BAR_COLORS = [
  '#6366f1',
  '#22c55e',
  '#f59e0b',
  '#a855f7',
  '#ef4444',
  '#14b8a6',
];

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

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs">
        <div className="text-gray-200 font-medium capitalize">{label}</div>
        <div className="text-indigo-400">{payload[0].value} annotations</div>
      </div>
    );
  }
  return null;
};

export default function LabelingPage() {
  const [projectId, setProjectId] = useState('1');
  const [autoSessionId, setAutoSessionId] = useState('');
  const [autoData, setAutoData] = useState<AutoLabel | null>(null);
  const [autoLoading, setAutoLoading] = useState(false);

  const [pushForm, setPushForm] = useState({
    session_id: '',
    project_title: 'VLA India',
  });
  const [pushResult, setPushResult] = useState('');

  const { data: stats, error, isLoading } = useSWR<LabelingStats>(
    `${API_BASE}/labeling/stats/${projectId}`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    { refreshInterval: 10000 }
  );

  const barData = Object.entries(stats?.by_action_type ?? {}).map(
    ([name, count]) => ({ name, count })
  );

  async function handleAutoLabel() {
    if (!autoSessionId.trim()) return;
    setAutoLoading(true);
    try {
      const r = await fetch(`${API_BASE}/labeling/auto/${autoSessionId}`);
      if (r.ok) setAutoData(await r.json());
    } catch {
      // ignore
    }
    setAutoLoading(false);
  }

  async function handlePush() {
    if (!pushForm.session_id.trim()) return;
    setPushResult('pushing...');
    try {
      const r = await fetch(`${API_BASE}/labeling/push/${pushForm.session_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_title: pushForm.project_title }),
      });
      if (r.ok) {
        const d = await r.json();
        setPushResult(`Pushed ${d.task_ids?.length ?? 0} tasks to project #${d.project_id}`);
      } else {
        setPushResult(`Error: ${r.status}`);
      }
    } catch {
      setPushResult('API offline');
    }
    setTimeout(() => setPushResult(''), 8000);
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Labeling</h1>
        <p className="text-sm text-gray-400 mt-1">
          LabelStudio integration and annotation tracking
        </p>
      </div>

      {error && <ErrorBanner />}

      {/* Project selector */}
      <div className="flex items-center gap-3">
        <label className="text-sm text-gray-400">LabelStudio Project ID:</label>
        <input
          type="text"
          value={projectId}
          onChange={(e) => setProjectId(e.target.value)}
          className="w-24 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
        />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <StatCard
          title="Total Annotated"
          value={isLoading ? '...' : stats?.total_annotated ?? 0}
          subtitle="labeled tasks"
          accent="green"
        />
        <StatCard
          title="Pending"
          value={isLoading ? '...' : stats?.pending ?? 0}
          subtitle="awaiting annotation"
          accent={
            (stats?.pending ?? 0) > 100
              ? 'yellow'
              : 'default'
          }
        />
        <StatCard
          title="Action Types"
          value={isLoading ? '...' : Object.keys(stats?.by_action_type ?? {}).length}
          subtitle="distinct actions"
          accent="indigo"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Action type distribution chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-200 mb-4">
            Action Type Distribution
          </h2>
          {isLoading ? (
            <div className="h-48 bg-gray-800 rounded animate-pulse" />
          ) : barData.length === 0 ? (
            <div className="h-48 flex items-center justify-center text-gray-500 text-sm">
              No annotation data available
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={barData} layout="vertical" margin={{ left: 10 }}>
                <XAxis
                  type="number"
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={80}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {barData.map((_, i) => (
                    <Cell
                      key={i}
                      fill={BAR_COLORS[i % BAR_COLORS.length]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Quick push form */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-200 mb-4">
            Push to LabelStudio
          </h2>
          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-400 block mb-1.5">Session ID *</label>
              <input
                type="text"
                value={pushForm.session_id}
                onChange={(e) =>
                  setPushForm((f) => ({ ...f, session_id: e.target.value }))
                }
                placeholder="e.g. session_20240115"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1.5">Project Title</label>
              <input
                type="text"
                value={pushForm.project_title}
                onChange={(e) =>
                  setPushForm((f) => ({ ...f, project_title: e.target.value }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
              />
            </div>
            {pushResult && (
              <div className="text-xs text-gray-300 bg-gray-800 rounded-lg px-3 py-2 font-mono">
                {pushResult}
              </div>
            )}
            <button
              onClick={handlePush}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <Tag className="w-4 h-4" />
              Push to LabelStudio
            </button>
          </div>
        </div>
      </div>

      {/* Sessions table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800">
          <h2 className="text-sm font-semibold text-gray-200">Session Annotation Status</h2>
        </div>
        {isLoading ? (
          <div className="p-6 space-y-2">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-10 bg-gray-800 rounded animate-pulse" />
            ))}
          </div>
        ) : !stats?.sessions || stats.sessions.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">
            No sessions with annotation data
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase tracking-wider">
                <th className="px-4 py-3 text-left">Session ID</th>
                <th className="px-4 py-3 text-right">Tasks Pushed</th>
                <th className="px-4 py-3 text-right">Done</th>
                <th className="px-4 py-3 text-left">Completion</th>
                <th className="px-4 py-3 text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {stats.sessions.map((s, idx) => (
                <tr
                  key={s.session_id}
                  className={clsx(
                    'border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors',
                    idx % 2 === 0 ? 'bg-transparent' : 'bg-gray-900/40'
                  )}
                >
                  <td className="px-4 py-3 font-mono text-indigo-400">{s.session_id}</td>
                  <td className="px-4 py-3 text-right text-gray-300">{s.tasks_pushed}</td>
                  <td className="px-4 py-3 text-right text-gray-300">{s.annotations_done}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={clsx(
                            'h-full rounded-full transition-all',
                            s.completion_pct >= 80
                              ? 'bg-green-500'
                              : s.completion_pct >= 40
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                          )}
                          style={{ width: `${s.completion_pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-10 text-right">
                        {s.completion_pct.toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <button className="text-xs px-3 py-1.5 bg-purple-600/20 hover:bg-purple-600 text-purple-400 hover:text-white rounded-lg transition-colors border border-purple-700">
                      Pull Annotations
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Auto-label preview */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-200 mb-4">
          Auto-Label Preview
        </h2>
        <div className="flex gap-3 mb-4">
          <input
            type="text"
            value={autoSessionId}
            onChange={(e) => setAutoSessionId(e.target.value)}
            placeholder="Enter session ID..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={handleAutoLabel}
            disabled={autoLoading}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {autoLoading ? <Loader className="w-4 h-4 animate-spin" /> : null}
            Fetch Labels
          </button>
        </div>
        {autoData && (
          <div className="space-y-4">
            {autoData.transcription && (
              <div>
                <div className="text-xs text-gray-400 mb-1.5">Whisper Transcription</div>
                <textarea
                  readOnly
                  value={autoData.transcription}
                  rows={3}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 font-mono resize-none focus:outline-none"
                />
              </div>
            )}
            {autoData.suggested_labels.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 mb-2">Suggested Labels</div>
                <div className="space-y-2">
                  {autoData.suggested_labels.map((lbl, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 bg-gray-800 rounded-lg px-3 py-2"
                    >
                      <span className="text-xs px-2 py-0.5 rounded bg-indigo-900/50 text-indigo-300 border border-indigo-700 capitalize">
                        {lbl.action_type}
                      </span>
                      <span className="flex-1 text-sm text-gray-300">
                        {lbl.instruction}
                      </span>
                      <span
                        className={clsx(
                          'text-xs font-mono font-medium',
                          lbl.confidence >= 0.8 ? 'text-green-400' : 'text-yellow-400'
                        )}
                      >
                        {(lbl.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

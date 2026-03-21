'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { clsx } from 'clsx';
import { format } from 'date-fns';
import { AlertTriangle, Play, Download, RefreshCw, Cpu } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { StatCard } from '@/components/StatCard';
import { API_BASE } from '@/lib/config';
import type { TrainingJob, Checkpoint, TrainingConfig } from '@/lib/types';

// Mock loss data for chart
const MOCK_LOSS: Array<{ epoch: number; loss: number; val_loss: number }> = Array.from(
  { length: 20 },
  (_, i) => ({
    epoch: i + 1,
    loss: parseFloat((2.5 * Math.exp(-0.15 * i) + 0.1 + Math.random() * 0.05).toFixed(3)),
    val_loss: parseFloat((2.8 * Math.exp(-0.13 * i) + 0.15 + Math.random() * 0.07).toFixed(3)),
  })
);

const HARDWARE_INFO = {
  gpu: 'NVIDIA RTX 4090',
  vram: '24 GB',
  disk: '2.1 TB free',
  ram: '64 GB',
};

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

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs">
        <div className="text-gray-400 mb-1">Epoch {label}</div>
        {payload.map((p) => (
          <div key={p.name} style={{ color: p.color }}>
            {p.name}: {p.value}
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function TrainingPage() {
  const [form, setForm] = useState<TrainingConfig>({
    dataset_dir: '/data/vla_dataset',
    epochs: 50,
    lr: 1e-4,
    batch_size: 8,
  });
  const [startResult, setStartResult] = useState('');

  const {
    data: trainingData,
    error,
    isLoading: trainLoading,
    mutate: mutateTrain,
  } = useSWR<{ active: TrainingJob | null; jobs: TrainingJob[] }>(
    `${API_BASE}/training/status`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    { refreshInterval: 5000 }
  );

  const {
    data: checkpoints,
    isLoading: ckptLoading,
    mutate: mutateCheckpoints,
  } = useSWR<Checkpoint[]>(
    `${API_BASE}/training/checkpoints`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    },
    { refreshInterval: 15000 }
  );

  const activeJob = trainingData?.active;

  async function handleStart() {
    setStartResult('starting...');
    try {
      const r = await fetch(`${API_BASE}/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (r.ok) {
        const d = await r.json();
        setStartResult(`Job started: ${d.job_id}`);
        mutateTrain();
      } else {
        setStartResult(`Error: ${r.status}`);
      }
    } catch {
      setStartResult('API offline');
    }
    setTimeout(() => setStartResult(''), 8000);
  }

  async function handleExport(name: string) {
    try {
      const r = await fetch(`${API_BASE}/training/export/${name}`, {
        method: 'POST',
      });
      if (r.ok) {
        const d = await r.json();
        alert(`Exported to HF: ${d.hf_url}`);
        mutateCheckpoints();
      }
    } catch {
      // ignore
    }
  }

  const epochProgress = activeJob
    ? Math.round((activeJob.epoch / activeJob.total_epochs) * 100)
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Training</h1>
          <p className="text-sm text-gray-400 mt-1">
            Monitor VLA model training and manage checkpoints
          </p>
        </div>
        <button
          onClick={() => { mutateTrain(); mutateCheckpoints(); }}
          className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-100 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {error && <ErrorBanner />}

      {/* Active job card */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-200">Active Training Job</h2>
          {activeJob && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-900/50 text-yellow-400 animate-pulse">
              Running
            </span>
          )}
        </div>

        {trainLoading ? (
          <div className="h-20 bg-gray-800 rounded animate-pulse" />
        ) : !activeJob ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            No active training job. Start one below.
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              title="Epoch"
              value={`${activeJob.epoch} / ${activeJob.total_epochs}`}
              subtitle={`${epochProgress}% complete`}
              accent="indigo"
            />
            <StatCard
              title="Current Loss"
              value={activeJob.loss.toFixed(4)}
              accent={activeJob.loss < 0.5 ? 'green' : activeJob.loss < 1.5 ? 'yellow' : 'red'}
            />
            <StatCard
              title="ETA"
              value={
                activeJob.eta_s
                  ? `${Math.floor(activeJob.eta_s / 60)}m`
                  : '—'
              }
              subtitle="estimated remaining"
            />
            <StatCard
              title="Status"
              value={activeJob.status}
              accent={activeJob.status === 'running' ? 'yellow' : 'default'}
            />
          </div>
        )}

        {activeJob && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Epoch progress</span>
              <span>{epochProgress}%</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                style={{ width: `${epochProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Loss chart */}
        <div className="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-200 mb-4">
            Loss Curve{' '}
            <span className="text-gray-500 font-normal text-xs">(prototype data)</span>
          </h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={MOCK_LOSS} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                label={{ value: 'Epoch', position: 'insideBottom', fill: '#6b7280', fontSize: 11 }}
              />
              <YAxis
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="loss"
                name="train_loss"
                stroke="#6366f1"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                name="val_loss"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
                strokeDasharray="4 2"
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-6 mt-2 justify-center">
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <div className="w-6 h-0.5 bg-indigo-500" />
              Train Loss
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <div className="w-6 h-0.5 bg-green-500 border-dashed" style={{ borderTop: '2px dashed #22c55e', background: 'none' }} />
              Val Loss
            </div>
          </div>
        </div>

        {/* Right column: start form + hardware */}
        <div className="space-y-4">
          {/* Start training form */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-sm font-semibold text-gray-200 mb-4">
              Start Training
            </h2>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 block mb-1">Dataset Dir</label>
                <input
                  type="text"
                  value={form.dataset_dir}
                  onChange={(e) => setForm((f) => ({ ...f, dataset_dir: e.target.value }))}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-100 font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Epochs</label>
                  <input
                    type="number"
                    value={form.epochs}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, epochs: parseInt(e.target.value) || 50 }))
                    }
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2 py-2 text-xs text-gray-100 focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">LR</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={form.lr}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, lr: parseFloat(e.target.value) || 1e-4 }))
                    }
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2 py-2 text-xs text-gray-100 focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Batch</label>
                  <input
                    type="number"
                    value={form.batch_size}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, batch_size: parseInt(e.target.value) || 8 }))
                    }
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2 py-2 text-xs text-gray-100 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>
              {startResult && (
                <div className="text-xs text-gray-300 bg-gray-800 rounded-lg px-3 py-2 font-mono">
                  {startResult}
                </div>
              )}
              <button
                onClick={handleStart}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors"
              >
                <Play className="w-4 h-4" />
                Start Training
              </button>
            </div>
          </div>

          {/* Hardware info */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-4 h-4 text-gray-400" />
              <h2 className="text-sm font-semibold text-gray-200">Hardware</h2>
              <span className="text-xs text-gray-500">(mock)</span>
            </div>
            <div className="space-y-2 text-sm">
              {Object.entries(HARDWARE_INFO).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-gray-400 capitalize">{k}</span>
                  <span className="text-gray-200 font-mono text-xs">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Checkpoints table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800">
          <h2 className="text-sm font-semibold text-gray-200">Checkpoints</h2>
        </div>
        {ckptLoading ? (
          <div className="p-6 space-y-2">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-10 bg-gray-800 rounded animate-pulse" />
            ))}
          </div>
        ) : !checkpoints || checkpoints.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">
            No checkpoints available
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase tracking-wider">
                <th className="px-4 py-3 text-left">Name</th>
                <th className="px-4 py-3 text-right">Epoch</th>
                <th className="px-4 py-3 text-right">Loss</th>
                <th className="px-4 py-3 text-left">Created</th>
                <th className="px-4 py-3 text-left">HF</th>
                <th className="px-4 py-3 text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((ckpt, idx) => (
                <tr
                  key={ckpt.name}
                  className={clsx(
                    'border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors',
                    idx % 2 === 0 ? 'bg-transparent' : 'bg-gray-900/40'
                  )}
                >
                  <td className="px-4 py-3 font-mono text-indigo-400 text-xs">{ckpt.name}</td>
                  <td className="px-4 py-3 text-right text-gray-300">{ckpt.epoch}</td>
                  <td
                    className={clsx(
                      'px-4 py-3 text-right font-mono font-medium',
                      ckpt.loss < 0.5
                        ? 'text-green-400'
                        : ckpt.loss < 1.0
                        ? 'text-yellow-400'
                        : 'text-red-400'
                    )}
                  >
                    {ckpt.loss.toFixed(4)}
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-xs">
                    {format(new Date(ckpt.created_at), 'MMM d HH:mm')}
                  </td>
                  <td className="px-4 py-3">
                    {ckpt.hf_pushed ? (
                      <span className="text-xs text-green-400">Pushed</span>
                    ) : (
                      <span className="text-xs text-gray-600">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => handleExport(ckpt.name)}
                      className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-green-600/20 hover:bg-green-600 text-green-400 hover:text-white rounded-lg transition-colors border border-green-700"
                    >
                      <Download className="w-3 h-3" />
                      Export to HF
                    </button>
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

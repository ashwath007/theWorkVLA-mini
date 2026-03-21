import { clsx } from 'clsx';
import { Check, X, Loader, Clock, SkipForward } from 'lucide-react';
import type { PipelineStage } from '@/lib/types';

interface StageProgressProps {
  stages: PipelineStage[];
}

const stageConfig: Record<string, { icon?: string }> = {
  assemble: {},
  sync: {},
  video: {},
  audio: {},
  imu: {},
  gps: {},
  transcribe: {},
  segment: {},
  chunk: {},
  validate: {},
  store: {},
};

const DEFAULT_STAGES: PipelineStage[] = [
  { name: 'assemble', status: 'pending' },
  { name: 'sync', status: 'pending' },
  { name: 'video', status: 'pending' },
  { name: 'audio', status: 'pending' },
  { name: 'imu', status: 'pending' },
  { name: 'gps', status: 'pending' },
  { name: 'transcribe', status: 'pending' },
  { name: 'segment', status: 'pending' },
  { name: 'chunk', status: 'pending' },
  { name: 'validate', status: 'pending' },
  { name: 'store', status: 'pending' },
];

function StatusIcon({ status }: { status: PipelineStage['status'] }) {
  if (status === 'done') return <Check className="w-3 h-3" />;
  if (status === 'error') return <X className="w-3 h-3" />;
  if (status === 'running') return <Loader className="w-3 h-3 animate-spin" />;
  if (status === 'skipped') return <SkipForward className="w-3 h-3" />;
  return <Clock className="w-3 h-3" />;
}

export function StageProgress({ stages }: StageProgressProps) {
  const displayStages =
    stages && stages.length > 0 ? stages : DEFAULT_STAGES;

  return (
    <div className="flex flex-wrap gap-2">
      {displayStages.map((stage, idx) => (
        <div key={stage.name} className="flex items-center">
          <div className="flex flex-col items-center gap-1">
            <div
              className={clsx(
                'w-8 h-8 rounded-full flex items-center justify-center border-2 transition-colors',
                stage.status === 'done' &&
                  'bg-green-500/20 border-green-500 text-green-400',
                stage.status === 'running' &&
                  'bg-yellow-500/20 border-yellow-500 text-yellow-400',
                stage.status === 'error' &&
                  'bg-red-500/20 border-red-500 text-red-400',
                stage.status === 'skipped' &&
                  'bg-gray-700 border-gray-600 text-gray-500',
                stage.status === 'pending' &&
                  'bg-gray-800 border-gray-700 text-gray-500'
              )}
            >
              <StatusIcon status={stage.status} />
            </div>
            <span className="text-xs text-gray-400 font-mono whitespace-nowrap">
              {stage.name}
            </span>
            {stage.elapsed_s !== undefined && stage.status === 'done' && (
              <span className="text-xs text-gray-600">
                {stage.elapsed_s.toFixed(1)}s
              </span>
            )}
          </div>
          {idx < displayStages.length - 1 && (
            <div
              className={clsx(
                'h-0.5 w-4 mx-1 mb-5',
                stage.status === 'done' ? 'bg-green-500' : 'bg-gray-700'
              )}
            />
          )}
        </div>
      ))}
    </div>
  );
}

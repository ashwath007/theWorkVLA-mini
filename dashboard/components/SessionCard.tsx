import Link from 'next/link';
import { clsx } from 'clsx';
import { Clock, Layers, Film } from 'lucide-react';
import type { Session } from '@/lib/types';

const scenarioBadgeColors: Record<string, string> = {
  delivery: 'bg-blue-900 text-blue-300 border-blue-700',
  driving: 'bg-green-900 text-green-300 border-green-700',
  warehouse: 'bg-orange-900 text-orange-300 border-orange-700',
  kitchen: 'bg-purple-900 text-purple-300 border-purple-700',
};

const statusColors: Record<string, string> = {
  done: 'text-green-400',
  running: 'text-yellow-400',
  error: 'text-red-400',
  pending: 'text-gray-400',
  new: 'text-gray-400',
};

interface SessionCardProps {
  session: Session;
}

export function SessionCard({ session }: SessionCardProps) {
  const badgeClass =
    scenarioBadgeColors[session.scenario] ??
    'bg-gray-800 text-gray-300 border-gray-600';
  const statusClass = statusColors[session.status] ?? 'text-gray-400';
  const durationH = (session.duration_s / 3600).toFixed(1);

  return (
    <Link href={`/sessions/${session.session_id}`}>
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 hover:border-indigo-600 transition-colors cursor-pointer">
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="text-sm font-mono text-gray-100 font-medium">
              {session.session_id}
            </div>
            <div
              className={clsx(
                'text-xs font-medium mt-0.5',
                statusClass
              )}
            >
              {session.status}
            </div>
          </div>
          <span
            className={clsx(
              'text-xs px-2 py-0.5 rounded border font-medium capitalize',
              badgeClass
            )}
          >
            {session.scenario}
          </span>
        </div>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-gray-800 rounded-lg p-2">
            <div className="flex items-center justify-center mb-1">
              <Layers className="w-3 h-3 text-gray-400" />
            </div>
            <div className="text-lg font-bold text-gray-100">{session.chunks}</div>
            <div className="text-xs text-gray-500">chunks</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-2">
            <div className="flex items-center justify-center mb-1">
              <Film className="w-3 h-3 text-gray-400" />
            </div>
            <div className="text-lg font-bold text-gray-100">{session.episodes}</div>
            <div className="text-xs text-gray-500">episodes</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-2">
            <div className="flex items-center justify-center mb-1">
              <Clock className="w-3 h-3 text-gray-400" />
            </div>
            <div className="text-lg font-bold text-gray-100">{durationH}</div>
            <div className="text-xs text-gray-500">hours</div>
          </div>
        </div>
      </div>
    </Link>
  );
}

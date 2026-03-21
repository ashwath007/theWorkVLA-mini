import { clsx } from 'clsx';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  accent?: 'default' | 'green' | 'yellow' | 'red' | 'indigo';
}

export function StatCard({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  accent = 'default',
}: StatCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
        {title}
      </div>
      <div
        className={clsx(
          'text-3xl font-bold mb-1',
          accent === 'green' && 'text-green-400',
          accent === 'yellow' && 'text-yellow-400',
          accent === 'red' && 'text-red-400',
          accent === 'indigo' && 'text-indigo-400',
          accent === 'default' && 'text-gray-100'
        )}
      >
        {value}
      </div>
      {(subtitle || trend) && (
        <div className="flex items-center gap-2">
          {subtitle && <span className="text-xs text-gray-500">{subtitle}</span>}
          {trend && trendValue && (
            <span
              className={clsx(
                'flex items-center gap-0.5 text-xs font-medium',
                trend === 'up' && 'text-green-400',
                trend === 'down' && 'text-red-400',
                trend === 'neutral' && 'text-gray-400'
              )}
            >
              {trend === 'up' && <TrendingUp className="w-3 h-3" />}
              {trend === 'down' && <TrendingDown className="w-3 h-3" />}
              {trendValue}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

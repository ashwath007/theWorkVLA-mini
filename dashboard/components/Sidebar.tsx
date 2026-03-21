'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Camera, Zap, Tag, Brain } from 'lucide-react';
import { clsx } from 'clsx';

const navItems = [
  { href: '/sessions', label: 'Sessions', icon: Camera },
  { href: '/pipeline', label: 'Pipeline', icon: Zap },
  { href: '/labeling', label: 'Labeling', icon: Tag },
  { href: '/training', label: 'Training', icon: Brain },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0 min-h-screen">
      <div className="h-14 flex items-center px-4 border-b border-gray-800">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-widest">
          Navigation
        </span>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                active
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-gray-800">
        <div className="text-xs text-gray-500">
          <div className="font-medium text-gray-400 mb-1">VLA Mini</div>
          <div>April 2026 Deadline</div>
        </div>
      </div>
    </aside>
  );
}

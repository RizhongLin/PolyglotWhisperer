import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  /** Progress value in the 0..1 range (e.g. 0.75 = 75 %). */
  value: number;
  indeterminate?: boolean;
}

export function Progress({ value, indeterminate, className, ...props }: ProgressProps) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className={cn('relative h-2 w-full overflow-hidden rounded-full bg-secondary', className)}
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      {...props}
    >
      <div
        className={cn(
          'h-full rounded-full bg-primary transition-[width] duration-300 ease-out',
          indeterminate && 'animate-pulse',
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

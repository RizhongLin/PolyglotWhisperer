import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

type Variant = 'default' | 'secondary' | 'outline' | 'destructive' | 'success' | 'warning';

const variants: Record<Variant, string> = {
  default: 'bg-primary text-primary-foreground border-transparent',
  secondary: 'bg-secondary text-secondary-foreground border-transparent',
  outline: 'text-foreground',
  destructive: 'bg-destructive text-destructive-foreground border-transparent',
  success: 'bg-success text-primary-foreground border-transparent',
  warning: 'bg-warning text-primary-foreground border-transparent',
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: Variant;
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium transition-colors',
        variants[variant],
        className,
      )}
      {...props}
    />
  );
}

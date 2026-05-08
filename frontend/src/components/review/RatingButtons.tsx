import { cn } from '@/lib/cn';
import type { FsrsRating } from '@/api/types';

interface RatingButtonsProps {
  onRate: (rating: FsrsRating) => void;
  disabled?: boolean;
}

const BUTTONS: Array<{
  rating: FsrsRating;
  label: string;
  hint: string;
  className: string;
}> = [
  {
    rating: 1,
    label: 'Again',
    hint: '<1m',
    className:
      'bg-destructive/10 text-destructive border-destructive/30 hover:bg-destructive/20',
  },
  {
    rating: 2,
    label: 'Hard',
    hint: '~10m',
    className:
      'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30 hover:bg-amber-500/20',
  },
  {
    rating: 3,
    label: 'Good',
    hint: '~1d',
    className:
      'bg-primary/10 text-primary border-primary/30 hover:bg-primary/20',
  },
  {
    rating: 4,
    label: 'Easy',
    hint: '~4d',
    className:
      'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20',
  },
];

export function RatingButtons({ onRate, disabled }: RatingButtonsProps) {
  return (
    <div className="grid grid-cols-4 gap-2">
      {BUTTONS.map((b) => (
        <button
          key={b.rating}
          type="button"
          disabled={disabled}
          onClick={() => onRate(b.rating)}
          className={cn(
            'flex flex-col items-center gap-0.5 rounded-md border px-3 py-2 text-sm font-medium transition-colors',
            'disabled:cursor-not-allowed disabled:opacity-50',
            b.className,
          )}
        >
          <span className="text-base">{b.label}</span>
          <span className="text-[10px] tabular-nums opacity-70">{b.hint}</span>
        </button>
      ))}
    </div>
  );
}

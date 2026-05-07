import { useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';
const KEY = 'pgw-theme';

function readPersisted(): Theme | null {
  const raw = localStorage.getItem(KEY);
  return raw === 'light' || raw === 'dark' ? raw : null;
}

function systemTheme(): Theme {
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function apply(theme: Theme): void {
  document.documentElement.classList.toggle('dark', theme === 'dark');
}

export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(() => readPersisted() ?? systemTheme());

  useEffect(() => {
    apply(theme);
  }, [theme]);

  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const handler = (): void => {
      if (!readPersisted()) setTheme(systemTheme());
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  return {
    theme,
    toggle: () => {
      setTheme((cur) => {
        const next: Theme = cur === 'dark' ? 'light' : 'dark';
        localStorage.setItem(KEY, next);
        return next;
      });
    },
  };
}

import { useEffect, useState } from 'react';
import { Link, useRouterState } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { LogOut, Moon, Settings2, ShieldCheck, Sun, UserCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTheme } from '@/lib/theme';
import { api } from '@/api/client';
import type { MeResponse } from '@/api/types';
import { cn } from '@/lib/cn';

const NAV = [
  { to: '/library', label: 'Library' },
  { to: '/studio', label: 'Studio' },
  { to: '/review', label: 'Review' },
] as const;

export function Header() {
  const { theme, toggle } = useTheme();
  const { location } = useRouterState();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const me = useQuery<MeResponse>({
    queryKey: ['me'],
    queryFn: () => api.me().catch(() => null as unknown as MeResponse),
    staleTime: 60_000,
    retry: false,
  });

  const user = me.data;

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 8);
    window.addEventListener('scroll', handler, { passive: true });
    return () => window.removeEventListener('scroll', handler);
  }, []);

  // Close dropdown on route change
  useEffect(() => setMenuOpen(false), [location.pathname]);

  const logout = async () => {
    try { await api.logout(); } catch { /* ignore */ }
    window.location.href = '/login';
  };

  const avatar = user?.email?.[0]?.toUpperCase() ?? '?';

  return (
    <header
      className={cn(
        'sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-shadow',
        scrolled && 'shadow-sm',
      )}
    >
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-6">
        <Link to="/" className="flex items-center gap-2 font-semibold">
          <img src="/icon.png" alt="" className="h-5 w-5 object-contain" />
          <span className="tracking-tight">PolyglotWhisperer</span>
        </Link>
        <nav className="flex items-center gap-1">
          {NAV.map((item) => {
            const active =
              location.pathname === item.to ||
              location.pathname.startsWith(`${item.to}/`);
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                  active
                    ? 'bg-secondary text-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-foreground',
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            aria-label={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
            onClick={toggle}
          >
            {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>

          {user ? (
            <div className="relative">
              <button
                type="button"
                onClick={() => setMenuOpen((v) => !v)}
                className="flex size-8 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary hover:bg-primary/20 transition-colors"
                aria-label="User menu"
              >
                {avatar}
              </button>
              {menuOpen ? (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setMenuOpen(false)} />
                  <div className="absolute right-0 top-full mt-1.5 z-20 w-56 rounded-lg border bg-card p-1 shadow-lg">
                    <div className="px-3 py-2 text-xs text-muted-foreground truncate">
                      {user.email}
                    </div>
                    <div className="border-t -mx-1 my-1" />
                    <Link
                      to="/settings"
                      className="flex items-center gap-2 rounded-md px-3 py-1.5 text-sm hover:bg-accent transition-colors"
                    >
                      <Settings2 className="size-4" /> Settings
                    </Link>
                    {user.is_admin ? (
                      <Link
                        to="/admin/users"
                        className="flex items-center gap-2 rounded-md px-3 py-1.5 text-sm hover:bg-accent transition-colors"
                      >
                        <ShieldCheck className="size-4" /> Admin
                      </Link>
                    ) : null}
                    <div className="border-t -mx-1 my-1" />
                    <button
                      type="button"
                      onClick={logout}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-sm hover:bg-accent transition-colors"
                    >
                      <LogOut className="size-4" /> Sign out
                    </button>
                  </div>
                </>
              ) : null}
            </div>
          ) : (
            <Link
              to="/login"
              className={cn(
                'flex size-8 items-center justify-center rounded-full bg-muted text-muted-foreground hover:bg-accent hover:text-foreground transition-colors',
              )}
              aria-label="Sign in"
            >
              <UserCircle className="size-4" />
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}

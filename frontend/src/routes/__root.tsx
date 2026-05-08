import { Outlet, createRootRouteWithContext, useNavigate, useLocation } from '@tanstack/react-router';
import { type QueryClient, useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { api } from '@/api/client';
import { Header } from '@/components/header';

interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootComponent,
});

const PUBLIC_PATHS = new Set(['/login', '/setup']);

function RootComponent() {
  const nav = useNavigate();
  const loc = useLocation();
  const isPublic = PUBLIC_PATHS.has(loc.pathname);

  // Cheap state probe — has_admin tells us whether to gate, authenticated
  // tells us if the visitor already has a session.
  const state = useQuery({
    queryKey: ['auth-state'],
    queryFn: () => api.authState(),
    // Re-evaluate aggressively so login/logout reflect immediately.
    staleTime: 0,
  });

  useEffect(() => {
    if (!state.data) return;
    if (isPublic) return;
    if (!state.data.has_admin) {
      nav({ to: '/setup', replace: true });
    } else if (!state.data.authenticated) {
      nav({ to: '/login', replace: true });
    }
  }, [state.data, isPublic, nav]);

  return (
    <div className="app-gradient flex min-h-screen flex-col">
      <Header />
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-6">
        <Outlet />
      </main>
      <footer className="border-t bg-background/60">
        <div className="mx-auto flex max-w-6xl flex-col items-center gap-3 px-6 py-6 text-xs text-muted-foreground">
          <img src="/logo.png" alt="PolyglotWhisperer" className="h-14 object-contain" />
          <span className="text-center">
            PolyglotWhisperer <span className="mx-1.5 opacity-40">·</span>{' '}
            <em>Video transcription &amp; translation for language learners</em>
          </span>
          <a
            href="https://github.com/RizhongLin/PolyglotWhisperer"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 rounded-full border border-border px-3 py-1 text-[11px] font-medium hover:border-primary hover:text-primary transition-colors no-underline"
          >
            GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}

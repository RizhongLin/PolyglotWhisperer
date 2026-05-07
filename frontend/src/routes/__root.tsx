import { Outlet, createRootRouteWithContext } from '@tanstack/react-router';
import { type QueryClient } from '@tanstack/react-query';
import { Header } from '@/components/header';

interface RouterContext {
  queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootComponent,
});

function RootComponent() {
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

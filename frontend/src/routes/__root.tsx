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
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4 text-xs text-muted-foreground">
          <span>PolyglotWhisperer · video transcription &amp; translation for language learners</span>
          <a
            href="https://github.com/RizhongLin/PolyglotWhisperer"
            target="_blank"
            rel="noreferrer"
            className="hover:text-foreground transition-colors"
          >
            GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}

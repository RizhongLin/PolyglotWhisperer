import { useQuery } from '@tanstack/react-query';
import { Link, createFileRoute } from '@tanstack/react-router';
import { Calendar, Clock, FileVideo, Languages, Sparkles } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { buttonClass } from '@/components/ui/button';
import { api } from '@/api/client';
import type { WorkspaceSummary } from '@/api/types';
import { formatDuration, formatUploadDate } from '@/lib/format';

export const Route = createFileRoute('/library/')({
  component: LibraryPage,
});

function LibraryPage() {
  const query = useQuery({
    queryKey: ['workspaces'],
    queryFn: () => api.workspaces(),
  });

  if (query.isLoading) {
    return <Skeleton />;
  }

  if (query.isError) {
    return (
      <Card className="p-6">
        <p className="text-sm text-destructive">
          Failed to load library: {(query.error as Error).message}
        </p>
      </Card>
    );
  }

  const items = query.data?.workspaces ?? [];

  if (items.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Library</h1>
          <p className="text-sm text-muted-foreground">
            {items.length} workspace{items.length === 1 ? '' : 's'}
          </p>
        </div>
        <Link to="/studio" className={buttonClass()}>
          <Sparkles className="size-4" />
          <span>New job</span>
        </Link>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {items.map((ws) => (
          <WorkspaceCard key={`${ws.slug}/${ws.timestamp}`} ws={ws} />
        ))}
      </div>
    </div>
  );
}

function WorkspaceCard({ ws }: { ws: WorkspaceSummary }) {
  const { data: languages } = useQuery({
    queryKey: ['languages'],
    queryFn: () => api.languages(),
    staleTime: Infinity,
  });
  const nameOf = (code: string | null) =>
    code ? (languages?.find((li) => li.code === code)?.name ?? code) : '';

  const target =
    ws.target_language && ws.language
      ? `${nameOf(ws.language)} → ${nameOf(ws.target_language)}`
      : nameOf(ws.language);
  return (
    <Link
      to="/library/$slug/$ts"
      params={{ slug: ws.slug, ts: ws.timestamp }}
      className="group"
    >
      <Card className="overflow-hidden p-0 transition-all hover:-translate-y-0.5 hover:shadow-md hover:border-primary/40">
        <div className="relative aspect-video w-full overflow-hidden bg-muted">
          {ws.thumbnail ? (
            <img
              src={ws.thumbnail}
              alt=""
              loading="lazy"
              className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-muted-foreground">
              <FileVideo className="size-10" />
            </div>
          )}
          {ws.duration ? (
            <span className="absolute bottom-2 right-2 rounded-md bg-black/70 px-2 py-0.5 text-xs font-medium text-white tabular-nums">
              {formatDuration(ws.duration)}
            </span>
          ) : null}
        </div>
        <div className="flex flex-col gap-2 p-3">
          <h3 className="line-clamp-2 text-sm font-semibold leading-snug">{ws.title}</h3>
          <div className="flex flex-wrap gap-1.5">
            {target ? (
              <Badge variant="secondary" className="font-mono">
                <Languages className="size-3" />
                {target}
              </Badge>
            ) : null}
            {ws.difficulty ? (
              <Badge variant="outline" className="font-mono">
                {ws.difficulty}
              </Badge>
            ) : null}
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {ws.upload_date ? (
              <span className="inline-flex items-center gap-1">
                <Calendar className="size-3" />
                {formatUploadDate(ws.upload_date)}
              </span>
            ) : null}
            {ws.uploader ? <span className="line-clamp-1">{ws.uploader}</span> : null}
          </div>
        </div>
      </Card>
    </Link>
  );
}

function EmptyState() {
  return (
    <Card className="flex flex-col items-center gap-4 p-12 text-center">
      <Clock className="size-10 text-muted-foreground" />
      <div>
        <h2 className="text-lg font-semibold">No workspaces yet</h2>
        <p className="text-sm text-muted-foreground">
          Process a video from the Studio page or via <code>pgw run</code>.
        </p>
      </div>
      <Link to="/studio" className={buttonClass()}>
        <Sparkles className="size-4" />
        <span>Open Studio</span>
      </Link>
    </Card>
  );
}

function Skeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="aspect-[3/4] animate-pulse rounded-xl border bg-muted/40"
        />
      ))}
    </div>
  );
}

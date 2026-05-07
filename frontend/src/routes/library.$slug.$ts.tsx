import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft,
  Braces,
  Brain,
  Calendar,
  Captions,
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Cpu,
  Download,
  ExternalLink,
  File,
  FileAudio,
  FileText,
  FileVideo,
  Languages,
  Loader2,
  BookOpen,
  RefreshCw,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button, buttonClass } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { api } from '@/api/client';
import type { SubtitleTrack, VocabSummary, VocabWord, WorkspaceDetail } from '@/api/types';
import { cn } from '@/lib/cn';
import { formatBytes, formatDuration, formatUploadDate } from '@/lib/format';
import {
  groupBilingualCues,
  parseVtt,
  splitBilingual,
  type VttCue,
} from '@/lib/vtt';

export const Route = createFileRoute('/library/$slug/$ts')({
  component: PlayerPage,
});

function PlayerPage() {
  const { slug, ts } = Route.useParams();
  const detail = useQuery({
    queryKey: ['workspace', slug, ts],
    queryFn: () => api.workspace(slug, ts),
  });
  const vocab = useQuery({
    queryKey: ['vocab', slug, ts],
    queryFn: () => api.vocab(slug, ts).catch(() => null),
  });

  if (detail.isLoading) return <div className="animate-pulse text-sm text-muted-foreground">Loading…</div>;
  if (detail.isError) {
    return (
      <Card className="p-6">
        <p className="text-sm text-destructive">
          Failed to load workspace: {(detail.error as Error).message}
        </p>
      </Card>
    );
  }
  if (!detail.data) {
    return (
      <Card className="p-6">
        <p className="text-sm text-destructive">
          Workspace data is empty — the API returned no content.
        </p>
      </Card>
    );
  }

  return (
    <PlayerLayout
      detail={detail.data}
      vocab={vocab.data ?? null}
      slug={slug}
      ts={ts}
    />
  );
}

interface LayoutProps {
  detail: WorkspaceDetail;
  vocab: VocabSummary | null;
  slug: string;
  ts: string;
}

function PlayerLayout({ detail, vocab, slug, ts }: LayoutProps) {
  const wsBase = `/ws/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}`;
  const videoSrc = detail.video ? `${wsBase}/${encodeURIComponent(detail.video)}` : null;
  const meta = detail.metadata;

  const pk = `pgw:track:${slug}/${ts}`;
  const tk = `pgw:time:${slug}/${ts}`;

  const [activeTrack, setActiveTrack] = useState<SubtitleTrack | null>(
    () => detail.tracks.find((t) => t.label === localStorage.getItem(pk)) ?? detail.tracks[0] ?? null,
  );
  const [cues, setCues] = useState<VttCue[]>([]);
  const [ttLoading, setTtLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const restoredRef = useRef(false);
  const syncingRef = useRef(false);

  // Persist selected track.
  useEffect(() => {
    if (activeTrack) localStorage.setItem(pk, activeTrack.label);
  }, [activeTrack, pk]);

  // Restore last playback position after video metadata loads.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || restoredRef.current) return;
    const handler = () => {
      const saved = localStorage.getItem(tk);
      if (saved != null) {
        const sec = Number(saved);
        if (Number.isFinite(sec) && sec > 1) v.currentTime = sec;
      }
      restoredRef.current = true;
    };
    if (v.readyState >= 1) {
      handler();
    } else {
      v.addEventListener('loadedmetadata', handler, { once: true });
    }
  }, [videoSrc, tk]);

  // Periodically persist playback position.
  useEffect(() => {
    const interval = setInterval(() => {
      const v = videoRef.current;
      if (!v || v.paused) return;
      localStorage.setItem(tk, String(v.currentTime));
    }, 5_000);
    return () => clearInterval(interval);
  }, [tk]);

  // Load VTT for the active track.
  useEffect(() => {
    if (!activeTrack) {
      setCues([]);
      setTtLoading(false);
      return;
    }
    let cancelled = false;
    setTtLoading(true);
    const url = `${wsBase}/${encodeURIComponent(activeTrack.file)}`;
    fetch(url)
      .then((r) => (r.ok ? r.text() : ''))
      .then((text) => {
        if (!cancelled) {
          setCues(parseVtt(text));
          setTtLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCues([]);
          setTtLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeTrack, wsBase]);

  // Toggle native <track> visibility in the <video> element.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    syncingRef.current = true;
    const id = requestAnimationFrame(() => {
      for (let i = 0; i < v.textTracks.length; i++) {
        const tt = v.textTracks[i]!;
        tt.mode = tt.label === activeTrack?.label ? 'showing' : 'hidden';
      }
      syncingRef.current = false;
    });
    return () => {
      cancelAnimationFrame(id);
      syncingRef.current = false;
    };
  }, [activeTrack]);

  // Sync activeTrack when the user uses the browser's built-in CC menu.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (!v.textTracks.addEventListener) return;
    const sync = () => {
      if (syncingRef.current) return;
      for (let i = 0; i < v.textTracks.length; i++) {
        const tt = v.textTracks[i]!;
        if (tt.mode === 'showing') {
          const match = detail.tracks.find((t) => t.label === tt.label);
          if (match) setActiveTrack(match);
          return;
        }
      }
    };
    v.textTracks.addEventListener('change', sync);
    return () => v.textTracks.removeEventListener('change', sync);
  }, [detail.tracks, videoSrc]);

  // Subscribe to <video> timeupdate (RAF-throttled) to drive transcript.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let rafId: number | null = null;
    const tick = () => {
      if (rafId) return;
      rafId = requestAnimationFrame(() => {
        setCurrentTime(v.currentTime);
        rafId = null;
      });
    };
    v.addEventListener('timeupdate', tick);
    return () => {
      v.removeEventListener('timeupdate', tick);
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [videoSrc]);

  const seekTo = (sec: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, sec - 0.05);
    void v.play();
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const v = videoRef.current;
      if (!v) return;
      const tag = document.activeElement?.tagName.toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;

      switch (e.key) {
        case 'ArrowLeft':
          v.currentTime = Math.max(0, v.currentTime - 5);
          break;
        case 'ArrowRight':
          v.currentTime = Math.min(v.duration || Infinity, v.currentTime + 5);
          break;
        case ' ': {
          const btn = (e.target as HTMLElement)?.closest('button, a, [role=button]');
          if (!btn) {
            e.preventDefault();
            v.paused ? v.play() : v.pause();
          }
          break;
        }
        case 'f':
          if (document.fullscreenElement) document.exitFullscreen();
          else v.requestFullscreen();
          break;
        case '1':
          v.playbackRate = 0.75;
          break;
        case '2':
          v.playbackRate = 1.0;
          break;
        case '3':
          v.playbackRate = 1.5;
          break;
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <Link to="/library" className={cn(buttonClass('ghost', 'sm'), 'self-start')}>
        <ArrowLeft className="size-4" />
        Library
      </Link>

      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {meta.title ?? slug}
          </h1>
          {meta.uploader ? (
            <p className="text-sm text-muted-foreground">{meta.uploader}</p>
          ) : null}
        </div>
        {meta.source_url ? (
          <a
            href={meta.source_url}
            target="_blank"
            rel="noreferrer"
            className={buttonClass('outline', 'sm')}
          >
            <ExternalLink className="size-3.5" /> Source
          </a>
        ) : null}
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="flex flex-col gap-4">
          <div className="overflow-hidden rounded-xl border bg-card shadow-sm">
            {videoSrc ? (
              <video
                ref={videoRef}
                src={videoSrc}
                controls
                className="aspect-video w-full bg-black"
                preload="metadata"
              >
                {detail.tracks.map((t) => (
                  <track
                    key={t.file}
                    kind="subtitles"
                    src={`${wsBase}/${encodeURIComponent(t.file)}`}
                    srcLang={t.lang}
                    label={t.label}
                    default={t === activeTrack}
                  />
                ))}
              </video>
            ) : (
              <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 bg-muted text-muted-foreground">
                <FileVideo className="size-10" />
                <span className="text-sm">Video file missing</span>
                {meta.source_url ? (
                  <RedownloadButton slug={slug} ts={ts} />
                ) : null}
              </div>
            )}
          </div>

          {detail.tracks.length > 0 ? (
            <Card className="p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Languages className="size-4 text-muted-foreground" />
                <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Track
                </span>
                {detail.tracks.map((t) => (
                  <button
                    key={t.file}
                    onClick={() => setActiveTrack(t)}
                    className={cn(
                      'rounded-md px-2.5 py-1 text-xs font-medium transition-colors',
                      activeTrack?.file === t.file
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-accent hover:text-foreground',
                    )}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </Card>
          ) : null}

          <Transcript
            cues={cues}
            activeTrack={activeTrack}
            ttLoading={ttLoading}
            currentTime={currentTime}
            onSeek={seekTo}
            vocab={vocab}
          />
        </div>

        <div className="flex flex-col gap-4">
          <DetailsCard detail={detail} />
          {vocab ? <VocabCard vocab={vocab} /> : null}
          <DownloadsCard detail={detail} wsBase={wsBase} />
          {detail.metadata.description ? (
            <Card>
              <CardHeader>
                <CardTitle>Description</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-line">
                  {detail.metadata.description}
                </p>
              </CardContent>
            </Card>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// ── Transcript ─────────────────────────────────────────────────────────

interface DisplayItem {
  start: number;
  end: number;
  primary: string;
  secondary: string;
}

function Transcript({
  cues,
  activeTrack,
  ttLoading,
  currentTime,
  onSeek,
  vocab,
}: {
  cues: VttCue[];
  activeTrack: SubtitleTrack | null;
  ttLoading: boolean;
  currentTime: number;
  onSeek: (s: number) => void;
  vocab: VocabSummary | null;
}) {
  const isBilingual =
    activeTrack?.label.toLowerCase().includes('bilingual') ?? false;

  const wordMap = useMemo(() => {
    const m = new Map<string, VocabWord>();
    if (!vocab?.top_rare_words) return m;
    for (const w of vocab.top_rare_words) {
      m.set(w.word.toLowerCase(), w);
      if (w.word !== w.word.toLowerCase()) m.set(w.word, w);
      if (w.lemma && w.lemma.toLowerCase() !== w.word.toLowerCase()) {
        m.set(w.lemma.toLowerCase(), w);
        if (w.lemma !== w.lemma.toLowerCase()) m.set(w.lemma, w);
      }
    }
    return m;
  }, [vocab]);

  const [popover, setPopover] = useState<{
    word: string;
    level: string;
    translation?: string;
    context?: string;
    x: number;
    y: number;
  } | null>(null);

  const displayItems: DisplayItem[] = useMemo(() => {
    if (isBilingual) {
      return groupBilingualCues(cues).filter(
        (p) => p.primary.trim() || p.secondary.trim(),
      );
    }
    return cues
      .map((c) => ({
        start: c.start,
        end: c.end,
        ...splitBilingual(c.text.replace(/<[^>]*>/g, '')),
      }))
      .filter((d) => d.primary.trim() || d.secondary.trim());
  }, [cues, isBilingual]);

  const ANTICIPATE = 0.3;
  const LINGER = 0.8;
  const activeIndex = useMemo(() => {
    for (let i = 0; i < displayItems.length; i++) {
      const item = displayItems[i]!;
      if (
        currentTime >= item.start - ANTICIPATE &&
        currentTime < item.end + LINGER
      ) {
        return i;
      }
    }
    return -1;
  }, [displayItems, currentTime]);

  const effectivePast = useMemo(() => {
    if (activeIndex >= 0) return activeIndex;
    // No active cue — dim everything whose end time has passed
    for (let i = displayItems.length - 1; i >= 0; i--) {
      if (currentTime >= displayItems[i]!.end + LINGER) return i;
    }
    return -1;
  }, [displayItems, currentTime, activeIndex]);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);

  // Scroll the transcript panel (only) when active cue leaves viewport.
  useEffect(() => {
    const el = activeRef.current;
    const container = scrollerRef.current;
    if (!el || !container) return;
    const topEdge = el.offsetTop - container.scrollTop;
    const margin = 48;
    if (topEdge < margin || topEdge + el.offsetHeight > container.clientHeight - margin) {
      const target = container.scrollTop + topEdge - container.clientHeight / 2 + el.offsetHeight / 2;
      container.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
    }
  }, [activeIndex]);

  const [copyToast, setCopyToast] = useState(false);
  const toastRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Dismiss popover on any click outside
  useEffect(() => {
    if (!popover) return;
    const close = () => setPopover(null);
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [popover]);

  if (ttLoading) {
    return (
      <Card className="flex items-center justify-center p-10">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
      </Card>
    );
  }

  if (displayItems.length === 0) {
    return (
      <Card className="p-6 text-center text-sm text-muted-foreground">
        No transcript loaded.
      </Card>
    );
  }

  const handleClick = (i: number, e: React.MouseEvent) => {
    // Vocab word-click popover detection
    const sel = window.getSelection();
    let word = sel?.toString().trim().toLowerCase() ?? '';
    if (!word) {
      const caretPos = (document as any).caretPositionFromPoint?.(e.clientX, e.clientY);
      if (caretPos?.offsetNode?.nodeType === Node.TEXT_NODE) {
        const text: string = caretPos.offsetNode.textContent ?? '';
        const start = text.lastIndexOf(' ', caretPos.offset) + 1;
        let end = text.indexOf(' ', caretPos.offset);
        if (end === -1) end = text.length;
        word = text.slice(start, end).trim().toLowerCase();
      } else {
        const range = (document as any).caretRangeFromPoint?.(e.clientX, e.clientY);
        if (range) {
          try { range.expand('word'); } catch { /* Firefox throws */ }
          word = range.toString().trim().toLowerCase();
        }
      }
    }
    const info = word && word.length >= 3 ? wordMap.get(word) : undefined;
    if (info) {
      setPopover({
        word: info.word,
        level: info.difficulty ?? (vocab?.estimated_difficulty ?? '?'),
        translation: info.translation,
        context: info.context,
        x: e.clientX,
        y: e.clientY,
      });
      e.nativeEvent.stopImmediatePropagation();
      return;
    }

    setPopover(null);

    if (i === activeIndex) {
      const item = displayItems[i]!;
      const text = item.secondary
        ? `${item.primary}\n${item.secondary}`
        : item.primary;
      navigator.clipboard.writeText(text).then(() => {
        setCopyToast(true);
        if (toastRef.current) clearTimeout(toastRef.current);
        toastRef.current = setTimeout(() => setCopyToast(false), 1200);
      });
    } else {
      onSeek(displayItems[i]!.start);
    }
  };

  return (
    <Card className="overflow-hidden p-0 relative">
      <CardHeader className="flex flex-row items-center justify-between border-b py-3">
        <CardTitle className="text-sm">Transcript</CardTitle>
        <span className="text-xs text-muted-foreground">
          {displayItems.length} segment{displayItems.length === 1 ? '' : 's'}
        </span>
      </CardHeader>
      <span
        className={cn(
          'absolute top-2 right-3 z-10 inline-flex items-center gap-1 rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary transition-all duration-200',
          copyToast ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-1 pointer-events-none',
        )}
        role="status"
        aria-live="polite"
      >
        <Check className="size-3" /> Copied
      </span>
      {popover ? (
        <div
          className="fixed z-50 max-w-[260px] rounded-lg border bg-card p-3 text-sm leading-relaxed shadow-lg pointer-events-none"
          style={{ left: Math.min(popover.x, window.innerWidth - 270), top: popover.y + 8 }}
        >
          <div className="font-semibold text-base">{popover.word}</div>
          <span
            className="inline-block mt-1 rounded-full px-2 py-px text-[11px] font-semibold text-white"
            style={{ backgroundColor: DIFFICULTY_COLORS[popover.level] ?? '#888' }}
          >
            {popover.level}
          </span>
          {popover.translation ? (
            <div className="mt-1.5 text-muted-foreground italic">{popover.translation}</div>
          ) : null}
          {popover.context ? (
            <div className="mt-1 text-xs text-muted-foreground">{popover.context}</div>
          ) : null}
        </div>
      ) : null}
      <div ref={scrollerRef} className="max-h-[28rem] overflow-y-auto p-2">
        {displayItems.map((item, i) => {
          const active = i === activeIndex;
          const past = activeIndex >= 0 ? i < activeIndex : i <= effectivePast;
          const future = activeIndex >= 0 ? i > activeIndex : i > effectivePast;
          return (
            <button
              key={i}
              ref={active ? activeRef : null}
              onClick={(e) => handleClick(i, e)}
              className={cn(
                'flex w-full flex-col gap-0.5 rounded-md px-3 py-2 text-left text-sm transition-[background-color,opacity,font-size,font-weight,border-left,padding-left]',
                active &&
                  'border-l-[3px] border-l-primary pl-[calc(0.75rem-3px)] bg-primary/10 text-foreground font-semibold text-[0.95rem]',
                past && 'opacity-65',
                future && 'opacity-40',
                !active && !past && !future && 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
              title={active ? 'Click to copy' : undefined}
            >
              <span className="font-medium leading-snug">{item.primary}</span>
              {item.secondary ? (
                <span className="text-xs leading-snug opacity-70">
                  {item.secondary}
                </span>
              ) : null}
              <span className="text-[10px] opacity-50 tabular-nums">
                {formatDuration(item.start)}
              </span>
            </button>
          );
        })}
      </div>
    </Card>
  );
}

// ── Cards ──────────────────────────────────────────────────────────────

function DetailsCard({ detail }: { detail: WorkspaceDetail }) {
  const { data: languages } = useQuery({
    queryKey: ['languages'],
    queryFn: () => api.languages(),
    staleTime: Infinity,
  });
  const nameOf = (code: string) =>
    languages?.find((li) => li.code === code)?.name ?? code;

  const m = detail.metadata;
  const rows: Array<[string, React.ReactNode, React.ReactNode]> = [];
  if (m.language && m.target_language) {
    rows.push([nameOf(m.language), <Languages className="size-3.5" key="l" />, `${nameOf(m.language)} → ${nameOf(m.target_language)}`]);
  } else if (m.language) {
    rows.push([nameOf(m.language), <Languages className="size-3.5" key="l" />, nameOf(m.language)]);
  }
  if (m.source_duration) rows.push(['Duration', <Clock className="size-3.5" key="d" />, formatDuration(m.source_duration)]);
  if (m.upload_date) rows.push(['Uploaded', <Calendar className="size-3.5" key="u" />, formatUploadDate(m.upload_date)]);
  if (m.whisper_model) rows.push(['Whisper', <Cpu className="size-3.5" key="w" />, <code>{m.whisper_model}</code>]);
  if (m.llm_model) rows.push(['LLM', <Brain className="size-3.5" key="b" />, <code>{m.llm_model}</code>]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Details</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
          {rows.map(([label, icon, value]) => (
            <div key={label} className="contents">
              <dt className="flex items-center gap-1.5 text-muted-foreground">{icon}{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}

const DIFFICULTY_COLORS: Record<string, string> = {
  A1: '#2e7d32',
  A2: '#558b2f',
  B1: '#f57f17',
  B2: '#e65100',
  C1: '#c62828',
  C2: '#6a1b9a',
};

function difficultyVariant(level: string): 'outline' | 'success' | 'warning' | 'destructive' {
  if (level.startsWith('A1')) return 'success';
  if (level.startsWith('A2')) return 'success';
  if (level.startsWith('B1')) return 'warning';
  if (level.startsWith('B2')) return 'warning';
  if (level.startsWith('C')) return 'destructive';
  return 'outline';
}

function VocabCard({ vocab }: { vocab: VocabSummary }) {
  const [expanded, setExpanded] = useState(false);
  const top = (vocab.top_rare_words ?? []).slice(0, expanded ? undefined : 8);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          Vocabulary
          {vocab.estimated_difficulty ? (
            <Badge variant={difficultyVariant(vocab.estimated_difficulty)} className="font-mono">
              {vocab.estimated_difficulty}
            </Badge>
          ) : null}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-2 text-center">
          <Stat label="Total" value={vocab.total_words} />
          <Stat label="Unique" value={vocab.unique_lemmas} />
          <Stat label="Rare" value={vocab.top_rare_words?.length} />
        </div>
        {top.length > 0 ? (
          <ul className="mt-4 flex flex-col divide-y">
            {top.map((w) => (
              <li key={w.word} className="flex items-center justify-between py-1.5 text-sm">
                <span className="font-medium">{w.word}</span>
                <span className="line-clamp-1 text-right text-xs text-muted-foreground">
                  {w.translation ?? ''}
                </span>
              </li>
            ))}
          </ul>
        ) : null}
        {(vocab.top_rare_words?.length ?? 0) > 8 ? (
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 w-full"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
            {expanded ? 'Show fewer' : `Show all ${vocab.top_rare_words?.length}`}
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="rounded-md bg-muted/50 p-2">
      <div className="text-xl font-semibold tabular-nums">{value ?? '—'}</div>
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
    </div>
  );
}

function fileIcon(suffix: string): React.ReactNode {
  const s = suffix.toLowerCase();
  if (['mp4', 'mkv', 'webm', 'mov', 'avi'].includes(s)) return <FileVideo className="size-3.5" />;
  if (['mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a'].includes(s)) return <FileAudio className="size-3.5" />;
  if (['vtt', 'srt'].includes(s)) return <Captions className="size-3.5" />;
  if (s === 'epub') return <BookOpen className="size-3.5" />;
  if (s === 'txt') return <FileText className="size-3.5" />;
  if (s === 'json') return <Braces className="size-3.5" />;
  if (['pdf', 'csv'].includes(s)) return <FileText className="size-3.5" />;
  return <File className="size-3.5" />;
}

function DownloadsCard({ detail, wsBase }: { detail: WorkspaceDetail; wsBase: string }) {
  if (detail.files.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="size-4" /> Downloads
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="flex flex-col divide-y">
          {detail.files.map((f) => (
            <li key={f.name} className="flex items-center justify-between py-1.5 text-sm">
              <a
                href={`${wsBase}/${encodeURIComponent(f.name)}`}
                download
                className="line-clamp-1 hover:underline flex items-center gap-1.5"
              >
                <span className="shrink-0 text-muted-foreground">{fileIcon(f.suffix)}</span>
                {f.name}
              </a>
              <span className="text-xs text-muted-foreground tabular-nums">
                {formatBytes(f.size)}
              </span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function RedownloadButton({ slug, ts }: { slug: string; ts: string }) {
  const [busy, setBusy] = useState(false);
  const [pct, setPct] = useState(0);
  const [status, setStatus] = useState('');
  const [detail, setDetail] = useState('');

  const STATUS_LABELS: Record<string, string> = {
    starting: 'Connecting…',
    downloading: 'Downloading…',
    processing: 'Processing…',
    done: 'Complete!',
    error: 'Failed',
  };

  const ctrlRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => ctrlRef.current?.abort();
  }, []);

  const fire = async () => {
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setBusy(true);
    setPct(0);
    setStatus('starting');
    setDetail('');
    try {
      const resp = await fetch(
        `/ws/${encodeURIComponent(slug)}/${encodeURIComponent(ts)}/redownload`,
        { method: 'POST', signal: ctrl.signal },
      );
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let lastData: Record<string, unknown> | null = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          const t = line.trim();
          if (!t) continue;
          try {
            lastData = JSON.parse(t) as Record<string, unknown>;
            setPct(Math.round((lastData.progress as number) || 0));
            setStatus((lastData.status as string) || '');
            setDetail((lastData.detail as string) || '');
          } catch {
            // skip malformed
          }
        }
      }
      if (lastData && lastData.status === 'error') {
        setStatus('error');
        setDetail((lastData.detail as string) || 'Unknown error');
      } else {
        setStatus('done');
        setDetail('Reloading…');
        setTimeout(() => window.location.reload(), 600);
      }
    } catch (err) {
      setStatus('error');
      setDetail((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-3">
      {busy ? (
        <div className="flex flex-col items-center gap-1">
          <div className="relative w-16 h-16">
            <svg viewBox="0 0 36 36" className="size-full -rotate-90">
              <circle
                cx="18" cy="18" r="15.9"
                fill="none"
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="3"
              />
              <circle
                cx="18" cy="18" r="15.9"
                fill="none"
                stroke="rgba(255,255,255,0.8)"
                strokeWidth="3"
                strokeDasharray={`${pct} 100`}
                strokeLinecap="round"
                className="transition-[stroke-dasharray] duration-300"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white tabular-nums">
              {pct}%
            </span>
          </div>
          <span className="text-xs font-medium text-white/80">{STATUS_LABELS[status] || status}</span>
          {detail ? <span className="text-[11px] text-white/50">{detail}</span> : null}
        </div>
      ) : (
        <Button onClick={fire} variant="outline" size="sm">
          <RefreshCw className="size-3.5" />
          {status === 'error' ? 'Try again' : 'Re-download'}
        </Button>
      )}
      {!busy && status === 'error' ? (
        <span className="text-xs text-destructive">{detail}</span>
      ) : null}
    </div>
  );
}

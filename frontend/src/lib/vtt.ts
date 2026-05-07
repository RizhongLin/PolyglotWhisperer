// Minimal WebVTT parser tuned for our subtitle output (single-line cues,
// HH:MM:SS.mmm timestamps, optional cue identifiers). Skips advanced
// features (regions, voice spans) — pgw only emits plain cues.

export interface VttCue {
  start: number; // seconds
  end: number; // seconds
  text: string;
}

const TIMESTAMP_RE =
  /(\d{2}:)?(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d{2}:)?(\d{2}):(\d{2})\.(\d{3})/;

function toSeconds(h: string | undefined, m: string, s: string, ms: string): number {
  const hh = h ? Number(h.replace(':', '')) : 0;
  return hh * 3600 + Number(m) * 60 + Number(s) + Number(ms) / 1000;
}

export function parseVtt(text: string): VttCue[] {
  const cues: VttCue[] = [];
  const blocks = text.replace(/\r\n/g, '\n').split(/\n\n+/);
  for (const block of blocks) {
    const lines = block.split('\n').filter((l) => l.trim() !== '');
    if (lines.length === 0) continue;
    if (lines[0]?.startsWith('WEBVTT')) continue;
    if (lines[0]?.startsWith('NOTE')) continue;
    const tsLine = lines.find((l) => TIMESTAMP_RE.test(l));
    if (!tsLine) continue;
    const m = TIMESTAMP_RE.exec(tsLine);
    if (!m) continue;
    const start = toSeconds(m[1], m[2]!, m[3]!, m[4]!);
    const end = toSeconds(m[5], m[6]!, m[7]!, m[8]!);
    const tsIndex = lines.indexOf(tsLine);
    const text_ = lines.slice(tsIndex + 1).join('\n').trim();
    if (text_) cues.push({ start, end, text: text_ });
  }
  return cues;
}

export interface VttCuePair {
  start: number;
  end: number;
  primary: string;
  secondary: string;
}

/**
 * Bilingual VTT: two-line cues where the top line is the translation and
 * the bottom is the original. This util splits a parsed bilingual cue
 * back into ``{primary, secondary}`` for our two-row transcript renderer.
 */
export function splitBilingual(text: string): { primary: string; secondary: string } {
  const idx = text.indexOf('\n');
  if (idx === -1) return { primary: text, secondary: '' };
  return {
    primary: text.slice(0, idx).trim(),
    secondary: text.slice(idx + 1).trim(),
  };
}

/**
 * Pair consecutive bilingual VTT cues that share identical timestamps.
 * The Python writer emits two cues per segment: odd-index = original,
 * even-index = translation. Pair them so the transcript renders both
 * languages per row instead of alternating single lines.
 */
export function groupBilingualCues(cues: VttCue[]): VttCuePair[] {
  const groups: VttCuePair[] = [];
  for (let i = 0; i < cues.length; i += 2) {
    const a = cues[i]!;
    const b = i + 1 < cues.length ? cues[i + 1] : null;
    if (b && a.start === b.start && a.end === b.end) {
      const ta = a.text.replace(/<[^>]*>/g, '').trim();
      const tb = b.text.replace(/<[^>]*>/g, '').trim();
      if (ta || tb) {
        groups.push({
          start: a.start,
          end: a.end,
          primary: tb,
          secondary: ta,
        });
      }
    } else {
      const ta = a.text.replace(/<[^>]*>/g, '').trim();
      if (ta) {
        groups.push({
          start: a.start,
          end: a.end,
          primary: ta,
          secondary: '',
        });
      }
      // Re-process b as start of next pair (old parity-preserving logic)
      if (b) i--;
    }
  }
  return groups;
}

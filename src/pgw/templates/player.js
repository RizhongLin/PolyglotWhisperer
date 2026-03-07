// --- Theme toggle ---
const root = document.documentElement;
const toggle = document.getElementById('theme-toggle');
const LIGHT = 'light', DARK = 'dark';
const prefersDark = matchMedia('(prefers-color-scheme: dark)');

function getEffective() {
  const saved = localStorage.getItem('pgw-theme');
  if (saved) return saved;
  return prefersDark.matches ? DARK : LIGHT;
}
function applyTheme(theme) {
  root.dataset.theme = theme;
  const icon = theme === DARK ? 'sun' : 'moon';
  toggle.innerHTML = `<i data-lucide="${icon}"></i>`;
  lucide.createIcons({ nodes: [toggle] });
}
applyTheme(getEffective());
toggle.addEventListener('click', () => {
  const next = root.dataset.theme === DARK ? LIGHT : DARK;
  localStorage.setItem('pgw-theme', next);
  applyTheme(next);
});
prefersDark.addEventListener('change', () => {
  if (!localStorage.getItem('pgw-theme')) applyTheme(getEffective());
});

// --- Helpers ---
const video = document.getElementById('player');
const tracks = video ? video.textTracks : [];

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

// --- Subtitle track toggles ---
const controls = document.getElementById('track-controls');
let _syncing = false;

for (let i = 0; i < tracks.length; i++) {
  const t = tracks[i];
  const id = 'track-' + i;
  const isDefault = (i === 0);
  const label = document.createElement('label');
  const chk = isDefault ? ' checked' : '';
  label.innerHTML =
    `<input type="checkbox" id="${id}"${chk}>` +
    `<span><i data-lucide="captions"></i>${t.label}</span>`;
  label.querySelector('input').addEventListener('change', (e) => {
    _syncing = true;
    if (e.target.checked) {
      // Radio behaviour: uncheck all others
      controls.querySelectorAll('input').forEach((inp) => {
        if (inp !== e.target) {
          inp.checked = false;
          tracks[parseInt(inp.id.split('-')[1])].mode = 'hidden';
        }
      });
      t.mode = 'showing';
    } else {
      t.mode = 'hidden';
    }
    _syncing = false;
    buildTranscript();
  });
  controls.appendChild(label);
  t.mode = isDefault ? 'showing' : 'hidden';
}

// Sync our toggles when the browser's built-in CC menu changes tracks
if (tracks.addEventListener) {
  tracks.addEventListener('change', () => {
    if (_syncing) return;
    controls.querySelectorAll('input').forEach((inp) => {
      const idx = parseInt(inp.id.split('-')[1]);
      inp.checked = (tracks[idx].mode === 'showing');
    });
    buildTranscript();
  });
}

// --- Transcript ---
const transcriptBody = document.getElementById('transcript-body');
const copyToast = document.getElementById('copy-toast');
let transcriptCues = [];
let activeRow = -1;
let toastTimer;

// Loose timing: anticipate before cue starts, linger after cue ends
const ANTICIPATE = 0.3;
const LINGER = 0.8;
const COPY_TOAST_MS = 1200;
const CUE_POLL_MS = 200;

function getFirstShowingTrack() {
  for (let i = 0; i < tracks.length; i++) {
    if (tracks[i].mode === 'showing') return tracks[i];
  }
  return null;
}

function isBilingualTrack(track) {
  return track && track.label && track.label.toLowerCase().includes('bilingual');
}

function groupBilingualCues(cues) {
  const groups = [];
  for (let i = 0; i < cues.length; i += 2) {
    const a = cues[i];
    const b = (i + 1 < cues.length) ? cues[i + 1] : null;
    if (b && a.startTime === b.startTime && a.endTime === b.endTime) {
      const ta = a.text.replace(/<[^>]*>/g, '').trim();
      const tb = b.text.replace(/<[^>]*>/g, '').trim();
      if (ta || tb) groups.push({ startTime: a.startTime, endTime: a.endTime, texts: [ta, tb] });
    } else {
      const ta = a.text.replace(/<[^>]*>/g, '').trim();
      if (ta) groups.push({ startTime: a.startTime, endTime: a.endTime, texts: [ta] });
      if (b) { i--; }  // re-process b as start of next pair
    }
  }
  return groups;
}

function buildTranscript() {
  const track = getFirstShowingTrack();
  if (!track || !track.cues || track.cues.length === 0) {
    transcriptBody.innerHTML =
      '<div class="transcript-empty">No transcript available</div>';
    transcriptCues = [];
    return;
  }
  transcriptCues = [];
  let html = '';
  const bilingual = isBilingualTrack(track);

  if (bilingual) {
    const groups = groupBilingualCues(track.cues);
    for (const g of groups) {
      transcriptCues.push(g);
      const idx = transcriptCues.length - 1;
      let textHtml = `<span class="cue-lang-a">${g.texts[0]}</span>`;
      if (g.texts[1]) textHtml += `<br><span class="cue-lang-b">${g.texts[1]}</span>`;
      html += `<div class="cue-row future" data-idx="${idx}" tabindex="0">`
        + `<span class="cue-time">${fmtTime(g.startTime)}</span>`
        + `<span class="cue-text">${textHtml}</span></div>`;
    }
  } else {
    for (let i = 0; i < track.cues.length; i++) {
      const cue = track.cues[i];
      const text = cue.text.replace(/<[^>]*>/g, '').trim();
      if (!text) continue;
      transcriptCues.push(cue);
      html += `<div class="cue-row future" data-idx="${transcriptCues.length - 1}" tabindex="0">`
        + `<span class="cue-time">${fmtTime(cue.startTime)}</span>`
        + `<span class="cue-text">${text}</span></div>`;
    }
  }

  transcriptBody.innerHTML = html;
  transcriptBody.querySelectorAll('.cue-row').forEach((row) => {
    row.addEventListener('click', () => {
      const idx = parseInt(row.dataset.idx);
      if (idx === activeRow) {
        // Active row: copy text
        const cueData = transcriptCues[idx];
        const text = cueData.texts
          ? cueData.texts.join('\n')
          : row.querySelector('.cue-text').textContent;
        navigator.clipboard.writeText(text).then(() => {
          copyToast.classList.add('show');
          clearTimeout(toastTimer);
          toastTimer = setTimeout(() => copyToast.classList.remove('show'), COPY_TOAST_MS);
        });
      } else {
        // Other rows: seek
        if (video) {
          video.currentTime = transcriptCues[idx].startTime;
          if (video.paused) video.play();
        }
      }
    });
    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        row.click();
      }
    });
  });
  highlightActiveCue();
}

function highlightActiveCue() {
  if (transcriptCues.length === 0) return;
  const t = video ? video.currentTime : 0;
  let idx = -1;
  // Binary search: find last cue where startTime - ANTICIPATE <= t
  let lo = 0, hi = transcriptCues.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (transcriptCues[mid].startTime - ANTICIPATE <= t) {
      idx = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (idx >= 0 && !(t < transcriptCues[idx].endTime + LINGER)) idx = -1;

  if (idx === activeRow) return;

  const rows = transcriptBody.querySelectorAll('.cue-row');

  // Update classes on all rows
  for (let i = 0; i < rows.length; i++) {
    rows[i].classList.remove('active', 'past', 'future');
    if (idx >= 0) {
      if (i < idx) rows[i].classList.add('past');
      else if (i === idx) rows[i].classList.add('active');
      else rows[i].classList.add('future');
    } else {
      // No active cue — mark all as past if before current time
      if (transcriptCues[i] && t >= transcriptCues[i].endTime) {
        rows[i].classList.add('past');
      } else {
        rows[i].classList.add('future');
      }
    }
  }

  // Auto-scroll to active row
  if (idx >= 0 && idx < rows.length) {
    const container = transcriptBody;
    const row = rows[idx];
    const rowTop = row.offsetTop - container.offsetTop;
    const target = rowTop - container.clientHeight / 2 + row.clientHeight / 2;
    container.scrollTop = target;
  }
  activeRow = idx;
}

let _rafId = null;
if (video) {
  video.addEventListener('timeupdate', () => {
    if (_rafId) return;
    _rafId = requestAnimationFrame(() => {
      highlightActiveCue();
      _rafId = null;
    });
  });
}

function waitForCues() {
  const track = getFirstShowingTrack();
  if (track && track.cues && track.cues.length > 0) {
    buildTranscript();
  } else {
    setTimeout(waitForCues, CUE_POLL_MS);
  }
}
waitForCues();

// --- Re-download with streaming progress ---
const RELOAD_DELAY_MS = 500;

window.redownload = async function() {
  const container = document.querySelector('.vm-content');
  if (!container) return;

  // Show progress UI with ring
  container.innerHTML =
    '<div class="vm-progress">' +
    '  <div class="vm-ring"><svg viewBox="0 0 36 36">' +
    '    <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="3"/>' +
    '    <circle class="vm-ring-fill" cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,0.8)"' +
    '      stroke-width="3" stroke-dasharray="0 100" stroke-linecap="round"' +
    '      transform="rotate(-90 18 18)"/>' +
    '  </svg><span class="vm-pct">0%</span></div>' +
    '  <p class="vm-status"><strong>Connecting\u2026</strong></p>' +
    '  <p class="vm-detail"></p>' +
    '</div>';

  const ring = container.querySelector('.vm-ring-fill');
  const pctEl = container.querySelector('.vm-pct');
  const statusEl = container.querySelector('.vm-status strong');
  const detailEl = container.querySelector('.vm-detail');

  const statusLabels = {
    starting: 'Connecting\u2026',
    downloading: 'Downloading\u2026',
    processing: 'Processing\u2026',
    done: 'Complete!',
    error: 'Failed',
  };

  function updateProgress(data) {
    const pct = Math.round(data.progress || 0);
    if (ring) ring.setAttribute('stroke-dasharray', pct + ' 100');
    if (pctEl) pctEl.textContent = pct + '%';
    if (statusEl) statusEl.textContent = statusLabels[data.status] || data.status;
    if (detailEl) detailEl.textContent = data.detail || '';
  }

  function showError(icon, title, detail) {
    container.innerHTML =
      '<i data-lucide="' + icon + '"></i>' +
      '<p><strong>' + title + '</strong></p>' +
      '<p>' + detail + '</p>' +
      '<button class="redownload-btn outline" onclick="redownload()">' +
      '<i data-lucide="refresh-cw"></i> Try again</button>';
    lucide.createIcons({ nodes: [container] });
  }

  try {
    const resp = await fetch(window.location.pathname + 'redownload', { method: 'POST' });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let lastData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();  // Keep incomplete line in buffer
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          lastData = JSON.parse(line);
          updateProgress(lastData);
        } catch(e) {}
      }
    }

    if (lastData && lastData.status === 'done') {
      setTimeout(() => window.location.reload(), RELOAD_DELAY_MS);
    } else if (lastData && lastData.status === 'error') {
      showError('circle-x', 'Download failed', lastData.detail || 'Unknown error');
    } else {
      // Stream ended without clear status — reload anyway
      window.location.reload();
    }
  } catch(e) {
    showError('wifi-off', 'Connection error', 'Could not reach the server.');
  }
};

// Render all Lucide icons
lucide.createIcons();

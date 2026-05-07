// Tiny typed DOM helpers — keeps the main entry point readable without
// pulling in a framework. The rendered HTML is server-side; this module
// only handles dynamic mutations (job cards, dialog open/close).

export function $<T extends HTMLElement = HTMLElement>(
  sel: string,
  root: ParentNode = document,
): T | null {
  return root.querySelector<T>(sel);
}

export function $$<T extends HTMLElement = HTMLElement>(
  sel: string,
  root: ParentNode = document,
): T[] {
  return Array.from(root.querySelectorAll<T>(sel));
}

export function escapeHtml(input: unknown): string {
  return String(input ?? '').replace(/[&<>"]/g, (c) => {
    switch (c) {
      case '&':
        return '&amp;';
      case '<':
        return '&lt;';
      case '>':
        return '&gt;';
      case '"':
        return '&quot;';
      default:
        return c;
    }
  });
}

export function readBool(form: HTMLFormElement, name: string): boolean {
  const el = form.elements.namedItem(name);
  return el instanceof HTMLInputElement && el.type === 'checkbox' ? el.checked : false;
}

export function readString(form: HTMLFormElement, name: string): string | null {
  const el = form.elements.namedItem(name);
  if (
    !(
      el instanceof HTMLInputElement ||
      el instanceof HTMLSelectElement ||
      el instanceof HTMLTextAreaElement
    )
  ) {
    return null;
  }
  const v = (el.value ?? '').trim();
  return v === '' ? null : v;
}

export function readNumber(form: HTMLFormElement, name: string): number | null {
  const v = readString(form, name);
  if (v === null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

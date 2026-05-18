import { useRef, useState } from 'react';
import { createFileRoute } from '@tanstack/react-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { KeyRound, Plus, Settings2, ShieldCheck, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { ApiError, api } from '@/api/client';
import type { Credential, Preferences } from '@/api/types';

export const Route = createFileRoute('/settings')({
  component: SettingsPage,
});

function SettingsPage() {
  const queryClient = useQueryClient();
  const me = useQuery({ queryKey: ['me'], queryFn: () => api.me(), staleTime: 60_000 });
  const credentials = useQuery({ queryKey: ['credentials'], queryFn: () => api.credentials() });
  const prefs = useQuery({ queryKey: ['preferences'], queryFn: () => api.preferences() });
  const languages = useQuery({ queryKey: ['languages'], queryFn: () => api.languages(), staleTime: Infinity });

  const langOptions = (languages.data ?? []).map((li) => (
    <option key={li.code} value={li.code}>{li.name} ({li.code})</option>
  ));

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary/10">
            <Settings2 className="size-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
            <p className="text-sm text-muted-foreground">
              Manage your account, defaults, and API credentials
            </p>
          </div>
        </div>
      </section>

      {/* ── Profile ── */}
      <ProfileCard
        email={me.data?.email ?? ''}
        isAdmin={me.data?.is_admin ?? false}
      />

      {/* ── Defaults ── */}
      <DefaultsCard
        prefs={prefs.data ?? {}}
        langOptions={langOptions}
        onSaved={() => { void queryClient.invalidateQueries({ queryKey: ['preferences'] }); }}
      />

      {/* ── Credentials ── */}
      <CredentialCard
        credentials={credentials.data ?? []}
        onChanged={() => { void queryClient.invalidateQueries({ queryKey: ['credentials'] }); }}
      />
    </div>
  );
}

function ProfileCard({ email, isAdmin }: { email: string; isAdmin: boolean }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ok, setOk] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const fd = new FormData(e.currentTarget as HTMLFormElement);
    const body = {
      current_password: fd.get('current_password') as string,
      new_password: fd.get('new_password') as string,
    };
    if (body.new_password !== fd.get('confirm_password')) {
      setError('Passwords do not match');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.changePassword(body);
      setOk(true);
      formRef.current?.reset();
      setTimeout(() => setOk(false), 3000);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="overflow-hidden">
      <div className="h-[3px] bg-linear-to-r from-primary/50 via-primary/20 to-transparent" />
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <KeyRound className="size-4 text-muted-foreground" /> Profile
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
          {email}
          {isAdmin ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
              <ShieldCheck className="size-3" /> Admin
            </span>
          ) : null}
        </div>
        <div className="border-t mb-4" />
        <form ref={formRef} onSubmit={onSubmit} className="flex flex-col gap-4 max-w-md">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="current_password">Current password</Label>
            <Input id="current_password" name="current_password" type="password" required />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="new_password">New password</Label>
            <Input id="new_password" name="new_password" type="password" required minLength={8} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="confirm_password">Confirm password</Label>
            <Input id="confirm_password" name="confirm_password" type="password" required minLength={8} />
          </div>
          {error ? <p className="text-sm text-destructive">{error}</p> : null}
          {ok ? <p className="text-sm text-success">Password changed</p> : null}
          <div className="flex justify-end">
            <Button type="submit" disabled={busy}>Change password</Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

function DefaultsCard({
  prefs,
  langOptions,
  onSaved,
}: {
  prefs: Preferences;
  langOptions: React.ReactNode;
  onSaved: () => void;
}) {
  const [busy, setBusy] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const fd = new FormData(e.currentTarget as HTMLFormElement);
    const v = (k: string) => {
      const x = fd.get(k);
      return typeof x === 'string' && x.trim() ? x.trim() : undefined;
    };
    setBusy(true);
    try {
      await api.updatePreferences({
        language: v('language'),
        translate: v('translate'),
        backend: v('backend'),
        llm_backend: v('llm_backend'),
      });
      onSaved();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="overflow-hidden">
      <div className="h-[3px] bg-linear-to-r from-primary/50 via-primary/20 to-transparent" />
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings2 className="size-4 text-muted-foreground" /> Defaults
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="s_language">Source language</Label>
              <Select id="s_language" name="language" defaultValue={prefs.language ?? ''}>
                <option value="">(config default)</option>
                {langOptions}
              </Select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="s_translate">Target language</Label>
              <Select id="s_translate" name="translate" defaultValue={prefs.translate ?? ''}>
                <option value="">(config default)</option>
                {langOptions}
              </Select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="s_backend">Whisper backend</Label>
              <Select id="s_backend" name="backend" defaultValue={prefs.backend ?? ''}>
                <option value="">(config default)</option>
                <option value="local">local</option>
                <option value="api">api</option>
              </Select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="s_llm_backend">LLM backend</Label>
              <Select id="s_llm_backend" name="llm_backend" defaultValue={prefs.llm_backend ?? ''}>
                <option value="">(config default)</option>
                <option value="local">local</option>
                <option value="api">api</option>
              </Select>
            </div>
          </div>
          <div className="flex justify-end">
            <Button type="submit" disabled={busy}>Save defaults</Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

function CredentialCard({
  credentials,
  onChanged,
}: {
  credentials: Credential[];
  onChanged: () => void;
}) {
  const whisperCreds = credentials.filter((c) => c.service === 'whisper');
  const llmCreds = credentials.filter((c) => c.service === 'llm');

  return (
    <Card className="overflow-hidden">
      <div className="h-[3px] bg-linear-to-r from-primary/50 via-primary/20 to-transparent" />
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <KeyRound className="size-4 text-muted-foreground" /> API Credentials
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        <CredentialGroup
          label="Whisper"
          creds={whisperCreds}
          service="whisper"
          onChanged={onChanged}
        />
        <CredentialGroup
          label="LLM"
          creds={llmCreds}
          service="llm"
          onChanged={onChanged}
        />
      </CardContent>
    </Card>
  );
}

function CredentialGroup({
  label,
  creds,
  service,
  onChanged,
}: {
  label: string;
  creds: Credential[];
  service: string;
  onChanged: () => void;
}) {
  const [adding, setAdding] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const fd = new FormData(e.currentTarget as HTMLFormElement);
    const v = (k: string) => {
      const x = fd.get(k);
      return typeof x === 'string' && x.trim() ? x.trim() : null;
    };
    const api_key = v('api_key');
    if (!api_key) return;
    setBusy(true);
    setError(null);
    try {
      await api.createCredential({
        service,
        provider: v('provider') || 'custom',
        api_key,
        api_base: v('api_base'),
        api_model: v('api_model'),
      });
      setAdding(false);
      onChanged();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const delete_ = async (id: number) => {
    await api.deleteCredential(id);
    onChanged();
  };

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <KeyRound className="size-3.5 text-muted-foreground" />
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{label}</span>
      </div>
      <div className="border-t mb-3" />

      {creds.length === 0 && !adding ? (
        <p className="text-sm text-muted-foreground">No {label} credentials saved.</p>
      ) : (
        <ul className="flex flex-col divide-y">
          {creds.map((c) => (
            <li key={c.id} className="flex items-center justify-between py-2 text-sm">
              <div>
                <span className="font-medium">{c.provider}</span>
                <span className="ml-2 text-muted-foreground font-mono text-xs">{c.masked_key}</span>
              </div>
              <button
                type="button"
                onClick={() => delete_(c.id)}
                className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                aria-label="Delete credential"
              >
                <Trash2 className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {adding ? (
        <form onSubmit={onSubmit} className="mt-3 grid grid-cols-1 gap-3 rounded-md border bg-muted/30 p-3 md:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={`${service}_provider`}>Provider</Label>
            <Input id={`${service}_provider`} name="provider" placeholder="groq, openai, deepseek…" required />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={`${service}_api_key`}>API key</Label>
            <Input id={`${service}_api_key`} name="api_key" required type="password" autoComplete="off" />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={`${service}_api_base`}>API base URL</Label>
            <Input id={`${service}_api_base`} name="api_base" placeholder="(optional)" />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={`${service}_api_model`}>Model</Label>
            <Input id={`${service}_api_model`} name="api_model" placeholder="(optional)" />
          </div>
          {error ? <p className="col-span-full text-sm text-destructive">{error}</p> : null}
          <div className="col-span-full flex justify-end gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => setAdding(false)}>Cancel</Button>
            <Button type="submit" size="sm" disabled={busy}>Save</Button>
          </div>
        </form>
      ) : (
        <Button type="button" variant="outline" size="sm" className="mt-2" onClick={() => setAdding(true)}>
          <Plus className="size-3.5" /> Add {label} credential
        </Button>
      )}
    </div>
  );
}

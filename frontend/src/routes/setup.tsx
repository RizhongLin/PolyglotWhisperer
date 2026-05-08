import { useState } from 'react';
import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError, ErrCode } from '@/api/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export const Route = createFileRoute('/setup')({
  component: SetupPage,
});

function SetupPage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (password !== confirm) {
      setErr('Passwords do not match.');
      return;
    }
    if (password.length < 8) {
      setErr('Password must be at least 8 characters.');
      return;
    }
    setPending(true);
    try {
      await api.setupAdmin(email, password);
      await qc.invalidateQueries();
      nav({ to: '/library' });
    } catch (e) {
      if (e instanceof ApiError && e.is(ErrCode.SetupAlreadyComplete)) {
        setErr('An admin already exists. Sign in instead.');
      } else if (e instanceof ApiError && e.is(ErrCode.AuthInvalidEmail)) {
        setErr('That email address is invalid.');
      } else {
        setErr((e as Error).message);
      }
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Create admin account</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4 text-sm text-muted-foreground">
            First-time setup: create the admin account that owns existing workspaces. You can
            invite more users later.
          </p>
          <form className="flex flex-col gap-4" onSubmit={onSubmit}>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="username"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="new-password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="confirm">Confirm password</Label>
              <Input
                id="confirm"
                type="password"
                autoComplete="new-password"
                required
                minLength={8}
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
              />
            </div>
            {err ? <p className="text-sm text-destructive">{err}</p> : null}
            <Button type="submit" disabled={pending}>
              {pending ? 'Creating account…' : 'Create admin'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

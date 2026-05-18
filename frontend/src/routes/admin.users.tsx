import { useState } from 'react';
import { createFileRoute } from '@tanstack/react-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { KeyRound, ShieldCheck, Trash2, UserPlus } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { ApiError, api } from '@/api/client';
import type { AdminUser } from '@/api/types';

export const Route = createFileRoute('/admin/users')({
  component: AdminUsersPage,
});

function AdminUsersPage() {
  const queryClient = useQueryClient();
  const users = useQuery({ queryKey: ['admin-users'], queryFn: () => api.adminUsers() });

  const refresh = () => { void queryClient.invalidateQueries({ queryKey: ['admin-users'] }); };

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary/10">
            <ShieldCheck className="size-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Users</h1>
            <p className="text-sm text-muted-foreground">
              Manage user accounts and credentials
            </p>
          </div>
        </div>
      </section>

      <Card className="overflow-hidden">
        <div className="h-[3px] bg-linear-to-r from-primary/50 via-primary/20 to-transparent" />
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="size-4 text-muted-foreground" /> All users
          </CardTitle>
          <CreateUserModal onCreated={refresh} />
        </CardHeader>
        <CardContent>
          {(users.data ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">No users found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2 font-medium">Email</th>
                    <th className="pb-2 font-medium">Admin</th>
                    <th className="pb-2 font-medium hidden md:table-cell">Created</th>
                    <th className="pb-2 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {(users.data ?? []).map((u) => (
                    <UserRow key={u.id} user={u} onChanged={refresh} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function UserRow({ user, onChanged }: { user: AdminUser; onChanged: () => void }) {
  const [resetting, setResetting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [newPw, setNewPw] = useState('');

  const resetPassword = async () => {
    if (!newPw || newPw.length < 8) return;
    setResetting(true);
    try {
      await api.adminResetPassword(user.id, { password: newPw });
      setNewPw('');
      setResetting(false);
    } catch {
      setResetting(false);
    }
  };

  const deleteUser = async () => {
    setDeleting(true);
    try {
      await api.adminDeleteUser(user.id);
      setConfirmDelete(false);
      onChanged();
    } catch {
      setDeleting(false);
    }
  };

  return (
    <tr className="text-sm">
      <td className="py-2 pr-4">{user.email}</td>
      <td className="py-2 pr-4">
        {user.is_admin ? (
          <Badge variant="default" className="text-[10px]">
            <ShieldCheck className="size-3" /> Admin
          </Badge>
        ) : null}
      </td>
      <td className="py-2 pr-4 text-muted-foreground hidden md:table-cell">
        {user.created_at?.slice(0, 10) ?? ''}
      </td>
      <td className="py-2 text-right">
        <div className="flex items-center justify-end gap-2">
          {resetting ? (
            <>
              <Input
                className="h-7 w-32 text-xs"
                placeholder="New password"
                type="password"
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') resetPassword(); }}
                autoFocus
              />
              <Button size="sm" variant="outline" className="h-7 text-xs" onClick={resetPassword}>Save</Button>
            </>
          ) : (
            <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={() => setResetting(true)}>Reset pw</Button>
          )}
          <button
            type="button"
            onClick={() => setConfirmDelete(true)}
            disabled={deleting}
            className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
            aria-label="Delete user"
          >
            <Trash2 className="size-3.5" />
          </button>
        </div>
      </td>
      <ConfirmDialog
        open={confirmDelete}
        title="Delete user"
        message={`Delete ${user.email} permanently? This cannot be undone.`}
        confirmLabel="Delete"
        destructive
        busy={deleting}
        onConfirm={deleteUser}
        onCancel={() => setConfirmDelete(false)}
      />
    </tr>
  );
}

function CreateUserModal({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const fd = new FormData(e.currentTarget as HTMLFormElement);
    const body = {
      email: (fd.get('email') as string)?.trim(),
      password: (fd.get('password') as string)?.trim(),
      is_admin: fd.get('is_admin') === 'on',
    };
    if (!body.email || !body.password || body.password.length < 8) return;
    setBusy(true);
    setError(null);
    try {
      await api.adminCreateUser(body);
      setOpen(false);
      onCreated();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Button size="sm" onClick={() => setOpen(true)}>
        <UserPlus className="size-3.5" /> Create user
      </Button>
      {open ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="w-full max-w-sm rounded-xl border bg-card p-6 shadow-lg">
            <h2 className="text-lg font-semibold mb-4">Create user</h2>
            <form onSubmit={onSubmit} className="flex flex-col gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="cu_email">Email</Label>
                <Input id="cu_email" name="email" type="email" required autoFocus />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="cu_password">Password</Label>
                <Input id="cu_password" name="password" type="password" required minLength={8} />
              </div>
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <Checkbox name="is_admin" /> Admin
              </label>
              {error ? <p className="text-sm text-destructive">{error}</p> : null}
              <div className="flex justify-end gap-2">
                <Button type="button" variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
                <Button type="submit" disabled={busy}>Create</Button>
              </div>
            </form>
          </div>
        </div>
      ) : null}
    </>
  );
}

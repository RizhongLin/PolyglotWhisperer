import { useQuery } from '@tanstack/react-query';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { api } from '@/api/client';

export type ExecutorChoice = 'auto' | 'worker' | 'server';

interface WorkerSelectProps {
  name?: string;
  defaultValue?: ExecutorChoice;
}

export function WorkerSelect({ name = 'executor', defaultValue = 'auto' }: WorkerSelectProps) {
  const { data: workers } = useQuery({
    queryKey: ['workers'],
    queryFn: () => api.workers(),
    staleTime: 15_000,
    refetchInterval: 20_000,
  });
  const connectedCount = workers?.filter((w) => w.connected).length ?? 0;

  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={name}>Run on</Label>
      <Select id={name} name={name} defaultValue={defaultValue}>
        <option value="auto">Auto (prefer worker)</option>
        <option value="worker">
          This machine
          {connectedCount > 0 ? ` (${connectedCount} connected)` : ' (none connected)'}
        </option>
        <option value="server">Server (admin-only)</option>
      </Select>
      {connectedCount === 0 ? (
        <p className="text-[11px] text-muted-foreground">
          Start a worker with{' '}
          <code className="bg-muted rounded px-1 py-px text-[10px]">pgw worker connect</code>
        </p>
      ) : null}
    </div>
  );
}

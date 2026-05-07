import { Outlet, createFileRoute } from '@tanstack/react-router';

// Section layout: just renders the matched child route. The grid lives
// at ``library.index.tsx``; the player at ``library.$slug.$ts.tsx``.
// Keeping this as a no-op layout means switching to a player URL takes
// over the full main area instead of nesting inside the grid.
export const Route = createFileRoute('/library')({
  component: LibraryLayout,
});

function LibraryLayout() {
  return <Outlet />;
}

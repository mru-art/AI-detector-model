import type { ReactNode } from "react";

export default function PageHeader({
  title,
  children,
  draft,
}: {
  title: string;
  children?: ReactNode;
  draft?: boolean;
}) {
  return (
    <header>
      {draft && (
        <p className="mb-3 inline-block rounded-md border border-ink px-3 py-1 text-xs font-bold uppercase tracking-widest">
          Draft
        </p>
      )}
      <h1 className="font-display text-6xl uppercase leading-none sm:text-8xl">{title}</h1>
      {children && <p className="mt-4 max-w-2xl text-base text-muted">{children}</p>}
    </header>
  );
}

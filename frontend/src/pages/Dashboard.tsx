import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, LayoutGrid, List, RefreshCw } from "lucide-react";

import {
  fetchDashboard,
  isTrained,
  summarize,
  type DashboardData,
  type ExperienceSummary,
  type Summary,
} from "@/lib/dashboard";

type View = "grid" | "list";

const pct = (n: number): string => `${n.toFixed(0)}%`;

function signedPoints(n: number): string {
  const sign = n > 0 ? "+" : n < 0 ? "−" : "";
  return `${sign}${Math.abs(n).toFixed(1)} pts`;
}

function MetricCard({ title, value, note }: { title: string; value: string; note: string }) {
  return (
    <div className="rounded-md bg-mist p-5">
      <p className="text-sm font-bold uppercase tracking-widest text-muted">{title}</p>
      <p className="mt-3 font-display text-6xl leading-none tabular-nums">{value}</p>
      <p className="mt-3 text-sm leading-relaxed text-muted">{note}</p>
    </div>
  );
}

function ExperienceGrid({ items }: { items: ExperienceSummary[] }) {
  return (
    <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {items.map((item, i) => {
        const newest = i === items.length - 1;
        return (
          <li key={item.id} className="rounded-md bg-mist p-6">
            <div className="flex items-start justify-between gap-4">
              <p className="font-display text-5xl leading-none text-muted">{item.id}</p>
              <p className="font-display text-5xl leading-none tabular-nums">{pct(item.now)}</p>
            </div>
            <h3 className="mt-4 text-xl font-bold">{item.generator}</h3>
            <div
              className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-line"
              role="img"
              aria-label={`${item.generator} accuracy now: ${pct(item.now)}`}
            >
              <div className="h-full rounded-full bg-ink" style={{ width: `${item.now}%` }} />
            </div>
            <p className="mt-3 text-sm text-muted">
              {newest
                ? `Newest · learned at ${pct(item.learned)}`
                : `Learned at ${pct(item.learned)} · now ${pct(item.now)} (${signedPoints(item.change)})`}
            </p>
          </li>
        );
      })}
    </ul>
  );
}

function ExperienceList({ items }: { items: ExperienceSummary[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-left">
        <thead>
          <tr className="bg-mist text-sm font-bold text-muted">
            <th scope="col" className="rounded-l-md px-5 py-3">
              Experience
            </th>
            <th scope="col" className="px-5 py-3">
              Generator
            </th>
            <th scope="col" className="px-5 py-3 text-right">
              Learned at
            </th>
            <th scope="col" className="px-5 py-3 text-right">
              Now
            </th>
            <th scope="col" className="rounded-r-md px-5 py-3 text-right">
              Change
            </th>
          </tr>
        </thead>
        <tbody className="group/body">
          {items.map((item) => (
            <tr
              key={item.id}
              className="transition-opacity group-hover/body:opacity-40 hover:opacity-100!"
            >
              <th scope="row" className="px-5 py-3.5 text-2xl font-bold text-muted sm:text-3xl">
                {item.id}
              </th>
              <td className="px-5 py-3.5 text-2xl font-bold sm:text-3xl">{item.generator}</td>
              <td className="px-5 py-3.5 text-right text-2xl font-bold tabular-nums sm:text-3xl">
                {pct(item.learned)}
              </td>
              <td className="px-5 py-3.5 text-right text-2xl font-bold tabular-nums sm:text-3xl">
                {pct(item.now)}
              </td>
              <td className="px-5 py-3.5 text-right text-2xl font-bold tabular-nums sm:text-3xl">
                {signedPoints(item.change)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Stronger cell = higher accuracy (darker in light mode, lighter in dark mode). */
function cellStyle(value: number): { background: string; light: boolean } {
  const t = Math.min(1, Math.max(0, (value - 50) / 50));
  const alpha = Math.round(8 + t * 82);
  return {
    background: `color-mix(in oklab, var(--ink) ${alpha}%, var(--paper))`,
    light: alpha >= 58,
  };
}

function Matrix({ data }: { data: DashboardData }) {
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] table-fixed border-separate border-spacing-1 text-center">
          <caption className="sr-only">
            Accuracy by generator after each training experience
          </caption>
          <colgroup>
            <col className="w-36" />
          </colgroup>
          <thead>
            <tr>
              <th scope="col" className="px-3 py-2 text-left text-sm font-bold text-muted">
                After…
              </th>
              {data.columns.map((c) => (
                <th key={c.id} scope="col" className="px-3 py-2 text-sm font-bold">
                  {c.label}
                  {c.kind === "heldout" && (
                    <span className="block text-xs font-normal text-muted">held out</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.matrix.map((row, r) => (
              <tr key={data.experiences[r].id}>
                <th scope="row" className="px-3 py-2 text-left text-sm font-bold">
                  {data.experiences[r].id}
                  <span className="block text-xs font-normal text-muted">
                    {data.experiences[r].generator}
                  </span>
                </th>
                {row.map((value, c) => {
                  const trained = isTrained(data, r, c);
                  const { background, light } = cellStyle(value);
                  return (
                    <td
                      key={data.columns[c].id}
                      style={{ background }}
                      className={`rounded border py-3.5 text-lg font-bold tabular-nums ${
                        light ? "text-paper" : "text-ink"
                      } ${trained ? "border-transparent" : "border-dashed border-muted"}`}
                    >
                      {pct(value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted">
        The stronger the cell, the higher the accuracy. Solid cells are generators the model has been trained on.
        Dashed cells are ones it hasn’t seen yet, so they show how well it generalizes.
      </p>
    </div>
  );
}

function Loading() {
  return (
    <div role="status" aria-busy="true" className="flex flex-col gap-6">
      <span className="sr-only">Loading dashboard…</span>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-40 animate-pulse rounded-md bg-mist" />
        ))}
      </div>
      <div className="h-72 animate-pulse rounded-md bg-mist" />
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("grid");
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const result = await fetchDashboard(controller.signal);
      if (controller.signal.aborted) return;
      setData(result);
    } catch (caught) {
      if (controller.signal.aborted) return;
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    return () => {
      abortRef.current?.abort();
    };
  }, [load]);

  const summary: Summary | null = useMemo(() => (data ? summarize(data) : null), [data]);

  return (
    <div className="flex flex-col gap-14">
      <header>
        <h1 className="font-display text-6xl uppercase leading-none sm:text-8xl">
          Continual learning
        </h1>
        <p className="mt-4 max-w-2xl text-base text-muted">
          How the detector holds up as it learns new image generators.
        </p>
      </header>

      {data?.isMock && (
        <div className="flex items-start gap-3 rounded-md border border-ink p-4 text-sm">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <p>
            <strong>Mock data.</strong> These numbers are made up to show the layout. They are not
            from the trained model.
          </p>
        </div>
      )}

      {loading && !data && <Loading />}

      {error && !data && (
        <div
          role="alert"
          className="flex flex-col items-start gap-4 rounded-md border border-ink p-6"
        >
          <div className="flex items-start gap-3">
            <AlertCircle className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
            <div>
              <p className="font-bold">Couldn’t load the dashboard</p>
              <p className="mt-1 text-sm text-muted">{error}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => void load()}
            className="inline-flex cursor-pointer items-center gap-2 rounded-md bg-ink px-5 py-3 text-base font-bold text-paper transition-opacity hover:opacity-85"
          >
            <RefreshCw className="size-4" aria-hidden="true" />
            Try again
          </button>
        </div>
      )}

      {data && summary && (
        <>
          <section aria-label="Summary" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Average accuracy"
              value={pct(summary.averageAccuracy)}
              note="Across real images and every generator trained so far."
            />
            <MetricCard
              title="Unseen generator"
              value={summary.heldOut ? pct(summary.heldOut.accuracy) : "—"}
              note={
                summary.heldOut
                  ? `Accuracy on ${summary.heldOut.label}, which the model never trained on.`
                  : "No held-out generator in this evaluation."
              }
            />
            <MetricCard
              title="Forgetting (BWT)"
              value={summary.bwt === null ? "—" : signedPoints(summary.bwt)}
              note="Average change on earlier generators. Negative means it forgot some."
            />
            <MetricCard
              title="Experiences"
              value={String(data.experiences.length)}
              note={data.experiences.map((e) => e.generator).join(", ")}
            />
          </section>

          <section aria-labelledby="experiences-title">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <h2
                id="experiences-title"
                className="font-display text-5xl uppercase leading-none sm:text-7xl"
              >
                Experiences
              </h2>
              <div className="inline-flex rounded-md bg-mist p-1" role="group" aria-label="View">
                {(
                  [
                    { id: "grid", label: "Grid", Icon: LayoutGrid },
                    { id: "list", label: "List", Icon: List },
                  ] as const
                ).map(({ id, label, Icon }) => (
                  <button
                    key={id}
                    type="button"
                    aria-pressed={view === id}
                    onClick={() => setView(id)}
                    className={`inline-flex cursor-pointer items-center gap-2 rounded px-4 py-2 text-base font-bold transition-colors ${
                      view === id ? "bg-ink text-paper" : "text-muted hover:text-ink"
                    }`}
                  >
                    <Icon className="size-4" aria-hidden="true" />
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-8">
              {view === "grid" ? (
                <ExperienceGrid items={summary.perExperience} />
              ) : (
                <ExperienceList items={summary.perExperience} />
              )}
            </div>
          </section>

          <section aria-labelledby="matrix-title">
            <h2
              id="matrix-title"
              className="font-display text-5xl uppercase leading-none sm:text-7xl"
            >
              Accuracy matrix
            </h2>
            <div className="mt-8">
              <Matrix data={data} />
            </div>
          </section>
        </>
      )}
    </div>
  );
}

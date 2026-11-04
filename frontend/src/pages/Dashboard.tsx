import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

import { fetchMetrics, perGenerator, type Metrics } from "@/lib/dashboard";
import { pct } from "@/lib/files";

type Measure = "ai_det" | "acc";
const MEASURES: { id: Measure; label: string; note: string }[] = [
  { id: "ai_det", label: "AI vs human", note: "Was a human image called human, and an AI image called AI?" },
  { id: "acc", label: "Exact source", note: "Did it name the exact source (human, or which generator)?" },
];

function signedPts(n: number): string {
  const v = n * 100;
  const sign = v > 0.05 ? "+" : v < -0.05 ? "−" : "";
  return `${sign}${Math.abs(v).toFixed(1)} pts`;
}

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="border-t border-line pt-4">
      <p className="text-[15px] text-muted">{label}</p>
      <p className="tabular mt-1 text-4xl font-extrabold tracking-tight sm:text-5xl">{value}</p>
      <p className="mt-2 text-sm text-muted">{note}</p>
    </div>
  );
}

/** Average accuracy after each experience: two series on one percentage axis. */
function TrendChart({ m }: { m: Metrics }) {
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const n = m.experiences.length;
  const series = [
    { id: "ai_det", label: "AI vs human", values: m.avg_ai_det, dash: "" },
    { id: "acc", label: "Exact source", values: m.avg_acc, dash: "6 5" },
  ];
  const all = series.flatMap((s) => s.values);
  const lo = Math.max(0, Math.floor((Math.min(...all) - 0.05) * 20) / 20);
  const W = 720;
  const H = 280;
  const pad = { l: 44, r: 150, t: 16, b: 52 };
  const x = (i: number) => pad.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - lo) / (1 - lo)) * (H - pad.t - pad.b);
  const ticks: number[] = [];
  for (let t = lo; t <= 1.0001; t += 0.05) ticks.push(Math.round(t * 100) / 100);

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const box = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    setHover(best);
  }

  return (
    <figure>
      <div className="-mx-5 overflow-x-auto px-5 sm:mx-0 sm:px-0">
      <div className="relative min-w-[600px]">
        <svg
          ref={ref}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full touch-pan-x"
          role="img"
          aria-label={`Average accuracy after each of ${n} experiences`}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth="1" />
              <text x={pad.l - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize="12" fill="var(--muted)" className="tabular">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {m.experiences.map((e, i) => (
            <g key={e.id}>
              <text x={x(i)} y={H - pad.b + 20} textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--ink)">
                {e.id}
              </text>
              <text x={x(i)} y={H - pad.b + 36} textAnchor="middle" fontSize="11" fill="var(--muted)">
                {e.label.length > 12 ? `${e.label.slice(0, 11)}…` : e.label}
              </text>
            </g>
          ))}
          {hover !== null && (
            <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={H - pad.b} stroke="var(--muted)" strokeWidth="1" strokeDasharray="3 3" />
          )}
          {series.map((s) => (
            <g key={s.id}>
              <polyline
                points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
                fill="none"
                stroke="var(--ink)"
                strokeWidth="2"
                strokeDasharray={s.dash}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {s.values.map((v, i) => (
                <circle key={i} cx={x(i)} cy={y(v)} r={hover === i ? 5 : 3.5} fill="var(--ink)" stroke="var(--surface)" strokeWidth="2" />
              ))}
              <text x={x(n - 1) + 12} y={y(s.values[n - 1])} dy="0.32em" fontSize="12.5" fill="var(--ink)" fontWeight="600">
                {s.label} {pct(s.values[n - 1])}
              </text>
            </g>
          ))}
        </svg>
        {hover !== null && (
          <div
            className="pointer-events-none absolute top-2 rounded-lg bg-ink px-3 py-2 text-sm text-paper shadow-lg"
            style={{ left: `${(x(hover) / W) * 100}%`, transform: hover > n / 2 ? "translateX(calc(-100% - 10px))" : "translateX(10px)" }}
          >
            <p className="font-semibold">
              After {m.experiences[hover].id}: {m.experiences[hover].label}
            </p>
            {series.map((s) => (
              <p key={s.id} className="tabular">
                {s.label}: {pct(s.values[hover], 1)}
              </p>
            ))}
          </div>
        )}
      </div>
      </div>
      <figcaption className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-muted">
        {series.map((s) => (
          <span key={s.id} className="inline-flex items-center gap-2">
            <svg width="28" height="8" aria-hidden="true">
              <line x1="0" x2="28" y1="4" y2="4" stroke="var(--ink)" strokeWidth="2" strokeDasharray={s.dash} />
            </svg>
            {s.label}
          </span>
        ))}
        <span>Average over every class learned so far.</span>
      </figcaption>
    </figure>
  );
}

/** Stronger cell = higher accuracy. Single hue, light to dark. */
function cellStyle(v: number): { background: string; light: boolean } {
  const t = Math.min(1, Math.max(0, (v - 0.5) / 0.5));
  const mix = Math.round(6 + t * 84);
  return { background: `color-mix(in oklab, var(--ink) ${mix}%, var(--surface))`, light: mix >= 55 };
}

function Matrix({ m, measure }: { m: Metrics; measure: Measure }) {
  const rows = m[measure];
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[680px] table-fixed border-separate border-spacing-1 text-center">
        <caption className="sr-only">
          {MEASURES.find((x) => x.id === measure)!.label} accuracy for each class after each experience
        </caption>
        <colgroup>
          <col className="w-40" />
        </colgroup>
        <thead>
          <tr>
            <th scope="col" className="px-2 py-2 text-left text-sm font-semibold text-muted">
              After
            </th>
            {m.classes.map((c) => (
              <th key={c.id} scope="col" className="px-1 py-2 align-bottom text-[13px] font-semibold leading-tight">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={m.experiences[r].id}>
              <th scope="row" className="px-2 py-1.5 text-left text-sm font-semibold">
                {m.experiences[r].id}
                <span className="block truncate text-xs font-normal text-muted">learned {m.experiences[r].label}</span>
              </th>
              {row.map((v, c) => {
                if (v === null) {
                  return (
                    <td key={c} className="rounded-md border border-dashed border-line text-sm text-muted">
                      <span className="sr-only">not learned yet</span>
                      <span aria-hidden="true">·</span>
                    </td>
                  );
                }
                const { background, light } = cellStyle(v);
                return (
                  <td
                    key={c}
                    style={{ background }}
                    title={`${m.classes[c].label} after ${m.experiences[r].id}: ${pct(v, 1)}`}
                    className={`tabular rounded-md py-3 text-[15px] font-semibold ${light ? "text-surface" : "text-ink"}`}
                  >
                    {Math.round(v * 100)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Retention({ m, measure }: { m: Metrics; measure: Measure }) {
  const items = perGenerator(m, measure);
  return (
    <ul className="divide-y divide-line border-y border-line">
      {items.map((it, i) => {
        const newest = i === items.length - 1;
        return (
          <li
            key={it.id}
            className="grid grid-cols-[2.5rem_1fr_auto] items-center gap-x-3 gap-y-2 py-3.5 sm:grid-cols-[3rem_14rem_1fr_9rem] sm:gap-x-4"
          >
            <span className="text-sm font-semibold text-muted">{it.experience}</span>
            <span className="truncate font-semibold">{it.label}</span>
            <div className="relative col-span-3 row-start-2 h-2 rounded-full bg-sunken sm:col-span-1 sm:col-start-3 sm:row-start-1">
              <div className="absolute inset-y-0 left-0 rounded-full bg-ink" style={{ width: `${it.now * 100}%` }} />
              <div
                className="absolute -top-1 h-4 w-0.5 bg-muted"
                style={{ left: `${it.learned * 100}%` }}
                title={`When learned: ${pct(it.learned, 1)}`}
              />
            </div>
            <span className="tabular col-start-3 row-start-1 text-right text-[15px] sm:col-start-4">
              <span className="font-semibold">{pct(it.now)}</span>{" "}
              <span className="text-muted">{newest ? "newest" : signedPts(it.change)}</span>
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<Metrics | null>(null);
  const [live, setLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [method, setMethod] = useState<string | null>(null);
  const [measure, setMeasure] = useState<Measure>("ai_det");
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async (m: string | null) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    try {
      const res = await fetchMetrics(m, controller.signal);
      if (controller.signal.aborted) return;
      setData(res.data);
      setLive(res.live);
    } catch (caught) {
      if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Something went wrong.");
    }
  }, []);

  useEffect(() => {
    void load(method);
    return () => abortRef.current?.abort();
  }, [load, method]);

  const s = data?.summary;
  const note = useMemo(() => MEASURES.find((x) => x.id === measure)!.note, [measure]);

  return (
    <div className="flex flex-col gap-16">
      <header className="flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-2xl">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">Model results</h1>
          <p className="mt-3 text-lg text-muted">
            The detector learned one generator at a time. This shows how accurate it was after each step, and how much it
            held on to what it had learned before.
          </p>
        </div>
        {data && data.methods.length > 1 && (
          <label className="flex flex-col gap-1.5 text-sm text-muted">
            Training method
            <select
              value={data.method}
              onChange={(e) => setMethod(e.target.value)}
              className="cursor-pointer rounded-full border border-line bg-surface px-4 py-2 text-[15px] font-semibold text-ink"
            >
              {data.methods.map((mm) => (
                <option key={mm.id} value={mm.id}>
                  {mm.label}
                  {mm.smoke ? " (quick test run)" : ""}
                </option>
              ))}
            </select>
          </label>
        )}
      </header>

      {error && !data && (
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-line bg-surface p-5">
          <AlertCircle className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
          <div className="flex-1">
            <p className="font-semibold">Couldn’t load the results</p>
            <p className="mt-1 text-sm text-muted">{error}</p>
          </div>
          <button type="button" onClick={() => void load(method)} className="inline-flex cursor-pointer items-center gap-2 rounded-full bg-ink px-4 py-2 font-semibold text-paper">
            <RefreshCw className="size-4" aria-hidden="true" /> Try again
          </button>
        </div>
      )}

      {!data && !error && (
        <div role="status" aria-busy="true" className="grid gap-6 sm:grid-cols-4">
          <span className="sr-only">Loading results…</span>
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="h-28 animate-pulse rounded-xl bg-sunken" />
          ))}
        </div>
      )}

      {data && s && (
        <>
          {(data.smoke_test || !live) && (
            <p className="-mt-8 text-sm text-muted">
              {data.smoke_test && <strong className="text-warn">Quick test run on a small sample, so these numbers are rough. </strong>}
              {!live && "Showing the results saved in the project, because the detector server couldn’t be reached."}
            </p>
          )}

          <section aria-label="Summary" className="grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="AI vs human accuracy"
              value={pct(s.final_real_vs_ai_accuracy ?? 0, 1)}
              note="On held-out test images from every class learned."
            />
            <Stat
              label="Names the exact source"
              value={pct(s.final_accuracy ?? 0, 1)}
              note={`Human, or which of ${data.experiences.length} generators.`}
            />
            <Stat
              label="ROC-AUC, AI vs human"
              value={(s.final_real_vs_ai_auc ?? 0).toFixed(3)}
              note="1.0 would mean every AI image scores above every human one."
            />
            <Stat
              label="Forgetting"
              value={`${((s.final_forgetting ?? 0) * 100).toFixed(1)} pts`}
              note="Average drop on older generators after learning new ones. Lower is better."
            />
          </section>

          <section aria-labelledby="trend-title" className="rounded-2xl border border-line bg-surface p-5 sm:p-8">
            <h2 id="trend-title" className="text-2xl font-bold tracking-tight">
              Accuracy after each new generator
            </h2>
            <p className="mt-1 text-muted">A flat line means new generators were learned without losing the old ones.</p>
            <div className="mt-6">
              <TrendChart m={data} />
            </div>
          </section>

          <section aria-labelledby="matrix-title">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h2 id="matrix-title" className="text-2xl font-bold tracking-tight">
                  Accuracy by class, step by step
                </h2>
                <p className="mt-1 text-muted">{note}</p>
              </div>
              <div className="inline-flex rounded-full bg-sunken p-1" role="group" aria-label="Measure">
                {MEASURES.map((mm) => (
                  <button
                    key={mm.id}
                    type="button"
                    aria-pressed={measure === mm.id}
                    onClick={() => setMeasure(mm.id)}
                    className={`cursor-pointer rounded-full px-4 py-1.5 text-sm transition-colors ${
                      measure === mm.id ? "bg-surface font-semibold shadow-sm" : "text-muted hover:text-ink"
                    }`}
                  >
                    {mm.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-6">
              <Matrix m={data} measure={measure} />
            </div>
            <p className="mt-3 text-sm text-muted">
              Percent correct. Darker cells are more accurate. Dotted cells are generators not learned yet at that step.
            </p>
          </section>

          <section aria-labelledby="retention-title">
            <h2 id="retention-title" className="text-2xl font-bold tracking-tight">
              Does it remember older generators?
            </h2>
            <p className="mt-1 text-muted">
              Accuracy now, compared with right after each one was learned (the tick mark).
            </p>
            <div className="mt-6">
              <Retention m={data} measure={measure} />
            </div>
          </section>
        </>
      )}
    </div>
  );
}

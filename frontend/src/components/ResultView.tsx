import { useState } from "react";
import { AlertTriangle, Eye, EyeOff } from "lucide-react";

import type { AnalysisResult } from "@/lib/analysis";

type Props = {
  url: string;
  fileName: string;
  result: AnalysisResult;
};

export default function ResultView({ url, fileName, result }: Props) {
  const [showFlags, setShowFlags] = useState(true);
  const isAi = result.verdict === "ai";
  const hasRegions = result.regions.length > 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(300px,1fr)]">
      <div className="flex items-center justify-center rounded-md bg-mist p-3">
        <div className="relative w-fit max-w-full">
          <img
            src={url}
            alt={`Uploaded image: ${fileName}`}
            className="block max-h-[70svh] w-auto max-w-full rounded-sm"
          />
          {showFlags && hasRegions && (
            <div className="pointer-events-none absolute inset-0" aria-hidden="true">
              {result.regions.map((r, i) => (
                <div
                  key={i}
                  className="absolute rounded-sm border-2 border-white bg-black/35 outline outline-2 outline-black"
                  style={{
                    left: `${r.x * 100}%`,
                    top: `${r.y * 100}%`,
                    width: `${r.w * 100}%`,
                    height: `${r.h * 100}%`,
                  }}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-5" aria-live="polite">
        {result.isMock && (
          <div className="flex items-start gap-3 rounded-md border border-ink p-4 text-sm">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            <p>
              <strong>Mock result.</strong> The detector isn’t connected yet, so this is not a real
              prediction.
            </p>
          </div>
        )}

        <div className="rounded-md bg-mist p-6">
          <p className="text-sm font-bold uppercase tracking-widest text-muted">Result</p>
          <h2 className="mt-2 font-display text-5xl uppercase leading-none sm:text-6xl">
            <span className="block">Likely</span>
            <span className="block whitespace-nowrap">{isAi ? "AI-generated" : "real"}</span>
          </h2>

          <dl className="mt-6 grid grid-cols-[auto_1fr] gap-x-6 gap-y-3 text-base">
            <dt className="text-muted">Confidence</dt>
            <dd className="text-right font-bold tabular-nums">
              {Math.round(result.confidence * 100)}%
            </dd>
            {result.likelySource && (
              <>
                <dt className="text-muted">Likely source</dt>
                <dd className="text-right font-bold">{result.likelySource}</dd>
              </>
            )}
            <dt className="text-muted">Flagged regions</dt>
            <dd className="text-right font-bold tabular-nums">{result.regions.length}</dd>
          </dl>

          {hasRegions && (
            <button
              type="button"
              aria-pressed={showFlags}
              onClick={() => setShowFlags((v) => !v)}
              className="mt-6 inline-flex w-full cursor-pointer items-center justify-center gap-2 rounded-md bg-paper px-4 py-3 text-base font-bold transition-colors hover:bg-line"
            >
              {showFlags ? (
                <EyeOff className="size-4" aria-hidden="true" />
              ) : (
                <Eye className="size-4" aria-hidden="true" />
              )}
              {showFlags ? "Hide flagged regions" : "Show flagged regions"}
            </button>
          )}
        </div>

        <p className="text-sm leading-relaxed text-muted">
          This is an estimate, not proof. Resizing and compression can weaken the signals a
          detector relies on, and results can be less reliable on generators it hasn’t seen.
        </p>
      </div>
    </div>
  );
}

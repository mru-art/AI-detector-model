import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, LoaderCircle, RefreshCw, X } from "lucide-react";

import Dropzone from "@/components/Dropzone";
import ResultView, { ArtworkStage } from "@/components/ResultView";
import { refreshHealth, useHealth, type HealthState } from "@/lib/api";
import { analyzeImage, takeHandOff, type AnalysisResult } from "@/lib/analysis";
import { formatSize, validateImage } from "@/lib/files";

type Phase =
  | { kind: "idle" }
  | { kind: "analyzing"; file: File; url: string }
  | { kind: "done"; file: File; url: string; result: AnalysisResult }
  | { kind: "error"; file: File; url: string; message: string };

function DetectorNotice({ state }: { state: HealthState }) {
  if (state.kind === "offline" || state.kind === "no-model") {
    const message =
      state.kind === "offline"
        ? "The detector can’t be reached right now. Free servers sleep when idle and can take a minute to wake up."
        : state.health.model.error ?? "The server is running, but no trained model is loaded.";
    return (
      <div role="status" className="flex items-start gap-3 rounded-xl border border-ai/30 bg-ai/8 p-4 text-[15px]">
        <AlertCircle className="mt-0.5 size-4 shrink-0 text-ai" aria-hidden="true" />
        <div className="flex-1">
          <p className="font-semibold">{state.kind === "offline" ? "Detector offline" : "No model loaded"}</p>
          <p className="mt-0.5 text-muted">{message}</p>
        </div>
        <button
          type="button"
          onClick={refreshHealth}
          className="cursor-pointer rounded-full px-3 py-1 text-sm font-semibold hover:bg-sunken"
        >
          Check again
        </button>
      </div>
    );
  }
  if (state.kind === "loading") {
    return (
      <div role="status" className="flex items-center gap-3 rounded-xl bg-sunken p-4 text-[15px]">
        <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
        The model is loading. Your image will be analyzed as soon as it’s ready.
      </div>
    );
  }
  return null;
}

export default function Analyze() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [notice, setNotice] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const health = useHealth();

  const urlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  useEffect(() => {
    if (phase.kind !== "analyzing") return;
    setElapsed(0);
    const id = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [phase.kind]);

  const run = useCallback(async (file: File, url: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setPhase({ kind: "analyzing", file, url });
    try {
      const result = await analyzeImage(file, controller.signal);
      if (!controller.signal.aborted) setPhase({ kind: "done", file, url, result });
    } catch (caught) {
      if (controller.signal.aborted) return;
      setPhase({ kind: "error", file, url, message: caught instanceof Error ? caught.message : "Something went wrong." });
      refreshHealth();
    }
  }, []);

  const accept = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (!file) return;
      const problem = validateImage(file);
      if (problem) {
        setNotice(problem);
        return;
      }
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      const url = URL.createObjectURL(file);
      urlRef.current = url;
      setNotice(files.length > 1 ? "One image at a time, so only the first one is being analyzed." : null);
      void run(file, url);
    },
    [run],
  );

  // A file dropped on the home page.
  useEffect(() => {
    const f = takeHandOff();
    if (f) accept([f]);
  }, [accept]);

  function reset() {
    abortRef.current?.abort();
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setNotice(null);
    setPhase({ kind: "idle" });
  }

  return (
    <div className="flex flex-col gap-8">
      {phase.kind !== "done" && (
        <header className="max-w-2xl">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">Analyze an artwork</h1>
          <p className="mt-3 text-lg text-muted">
            Upload a painting, drawing, illustration or photo. You’ll get an estimate of whether it was made by a person
            or an AI model, and which areas look machine-made.
          </p>
        </header>
      )}

      {phase.kind === "idle" && <DetectorNotice state={health} />}

      {notice && (
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-line bg-surface p-4 text-[15px]">
          <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <p className="flex-1">{notice}</p>
          <button
            type="button"
            aria-label="Dismiss message"
            onClick={() => setNotice(null)}
            className="cursor-pointer rounded-full p-1 hover:bg-sunken"
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        </div>
      )}

      {phase.kind === "idle" && (
        <>
          <Dropzone onFiles={accept} />
          <p className="max-w-2xl text-sm text-muted">
            For the most reliable result, use the original file. Screenshots, resized copies and images re-saved by
            social media lose the fine detail the detector relies on.
          </p>
        </>
      )}

      {(phase.kind === "analyzing" || phase.kind === "error") && (
        <section className="grid gap-8 lg:grid-cols-[minmax(0,1.55fr)_minmax(320px,1fr)]">
          <ArtworkStage url={phase.url} alt={`Preview of ${phase.file.name}`} scanning={phase.kind === "analyzing"} />

          <div className="flex flex-col gap-5 rounded-2xl border border-line bg-surface p-6 lg:self-start">
            <div className="min-w-0">
              <p className="break-all text-lg font-bold">{phase.file.name}</p>
              <p className="text-sm text-muted">{formatSize(phase.file.size)}</p>
            </div>

            {phase.kind === "analyzing" && (
              <>
                <div role="status" aria-live="polite">
                  <p className="flex items-center gap-2 font-semibold">
                    <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
                    Checking the image… <span className="tabular text-muted">{elapsed}s</span>
                  </p>
                  <p className="mt-2 text-sm text-muted">
                    {elapsed < 12
                      ? "The model looks at several patches of the original pixels. This usually takes a few seconds."
                      : "Still working. If the server was asleep, the first image can take up to a minute."}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={reset}
                  className="cursor-pointer rounded-full border border-line px-5 py-2.5 font-semibold transition-colors hover:bg-sunken"
                >
                  Cancel
                </button>
              </>
            )}

            {phase.kind === "error" && (
              <>
                <div role="alert" className="flex items-start gap-3 rounded-xl bg-ai/8 p-4 text-[15px]">
                  <AlertCircle className="mt-0.5 size-4 shrink-0 text-ai" aria-hidden="true" />
                  <div>
                    <p className="font-semibold">Couldn’t analyze this image</p>
                    <p className="mt-1 text-muted">{phase.message}</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => run(phase.file, phase.url)}
                  className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-full bg-ink px-5 py-3 font-semibold text-paper transition-opacity hover:opacity-85"
                >
                  <RefreshCw className="size-4" aria-hidden="true" />
                  Try again
                </button>
                <button
                  type="button"
                  onClick={reset}
                  className="cursor-pointer rounded-full border border-line px-5 py-2.5 font-semibold transition-colors hover:bg-sunken"
                >
                  Choose another image
                </button>
              </>
            )}
          </div>
        </section>
      )}

      {phase.kind === "done" && (
        <ResultView url={phase.url} fileName={phase.file.name} result={phase.result} onReset={reset} />
      )}
    </div>
  );
}

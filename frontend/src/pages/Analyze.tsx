import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { AlertCircle, ImagePlus, LoaderCircle, RefreshCw, Replace, Trash2, X } from "lucide-react";

import ResultView from "@/components/ResultView";
import { analyzeImage, type AnalysisResult } from "@/lib/analysis";
import { ALLOWED_LABEL, ALLOWED_TYPES, MAX_MB, formatSize, validateImage } from "@/lib/files";

type Phase =
  | { kind: "idle" }
  | { kind: "ready"; file: File; url: string }
  | { kind: "analyzing"; file: File; url: string }
  | { kind: "done"; file: File; url: string; result: AnalysisResult }
  | { kind: "error"; file: File; url: string; message: string };

const buttonBase =
  "inline-flex cursor-pointer items-center justify-center gap-2 rounded-md px-5 py-3 text-base font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-50";
const primary = `${buttonBase} bg-ink text-paper hover:opacity-85`;
const secondary = `${buttonBase} bg-mist hover:bg-line`;

export default function Analyze() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const inputRef = useRef<HTMLInputElement>(null);
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

  function accept(files: File[]) {
    const file = files[0];
    if (!file) return;
    const problem = validateImage(file);
    if (problem) {
      setNotice(problem);
      return;
    }
    abortRef.current?.abort();
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    const url = URL.createObjectURL(file);
    urlRef.current = url;
    setNotice(files.length > 1 ? "Only one image at a time. Using the first one." : null);
    setPhase({ kind: "ready", file, url });
  }

  function reset() {
    abortRef.current?.abort();
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setNotice(null);
    setPhase({ kind: "idle" });
  }

  async function run(file: File, url: string) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setNotice(null);
    setPhase({ kind: "analyzing", file, url });
    try {
      const result = await analyzeImage(file, controller.signal);
      if (controller.signal.aborted) return;
      setPhase({ kind: "done", file, url, result });
    } catch (caught) {
      if (controller.signal.aborted) return;
      setPhase({
        kind: "error",
        file,
        url,
        message: caught instanceof Error ? caught.message : "Something went wrong.",
      });
    }
  }

  function cancel() {
    if (phase.kind !== "analyzing") return;
    abortRef.current?.abort();
    setPhase({ kind: "ready", file: phase.file, url: phase.url });
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    accept(Array.from(event.dataTransfer.files));
  }

  const hiddenInput = (
    <input
      ref={inputRef}
      type="file"
      accept={ALLOWED_TYPES.join(",")}
      className="sr-only"
      tabIndex={-1}
      onChange={(e) => {
        accept(Array.from(e.target.files ?? []));
        e.target.value = "";
      }}
    />
  );

  return (
    <div className="flex flex-col gap-8">
      <header>
        <h1 className="font-display text-6xl uppercase leading-none sm:text-8xl">
          Analyze an image
        </h1>
        <p className="mt-4 max-w-xl text-base text-muted">
          Upload one image to estimate whether it is real or AI-generated.
        </p>
      </header>

      {notice && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-md border border-ink p-4 text-sm"
        >
          <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <p className="flex-1">{notice}</p>
          <button
            type="button"
            aria-label="Dismiss message"
            onClick={() => setNotice(null)}
            className="cursor-pointer rounded p-0.5 hover:bg-mist"
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        </div>
      )}

      {phase.kind === "idle" && (
        <section
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex min-h-[44svh] flex-col items-center justify-center gap-5 rounded-md border-2 border-dashed p-8 text-center transition-colors ${
            dragging ? "border-ink bg-line" : "border-line bg-mist"
          }`}
        >
          <ImagePlus className="size-10" aria-hidden="true" />
          <div>
            <p className="font-display text-4xl uppercase sm:text-5xl">Drop an image here</p>
            <p className="mt-2 text-base text-muted">or choose one from your device</p>
          </div>
          <button type="button" className={primary} onClick={() => inputRef.current?.click()}>
            Choose image
          </button>
          <p className="text-sm text-muted">
            {ALLOWED_LABEL} · up to {MAX_MB} MB · one image at a time
          </p>
          {hiddenInput}
        </section>
      )}

      {(phase.kind === "ready" || phase.kind === "analyzing" || phase.kind === "error") && (
        <section className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(300px,1fr)]">
          <div className="flex items-center justify-center rounded-md bg-mist p-3">
            <img
              src={phase.url}
              alt={`Preview of ${phase.file.name}`}
              className="block max-h-[70svh] w-auto max-w-full rounded-sm"
            />
          </div>

          <div className="flex flex-col gap-4 rounded-md bg-mist p-6">
            <div className="min-w-0">
              <p className="text-sm font-bold uppercase tracking-widest text-muted">
                {phase.kind === "analyzing" ? "Analyzing" : "Selected image"}
              </p>
              <p className="mt-1 break-all text-lg font-bold">{phase.file.name}</p>
              <p className="text-sm text-muted">{formatSize(phase.file.size)}</p>
            </div>

            {phase.kind === "ready" && (
              <>
                <button type="button" className={primary} onClick={() => run(phase.file, phase.url)}>
                  Analyze image
                </button>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    className={`${secondary} !bg-paper hover:!bg-line`}
                    onClick={() => inputRef.current?.click()}
                  >
                    <Replace className="size-4" aria-hidden="true" />
                    Replace
                  </button>
                  <button
                    type="button"
                    className={`${secondary} !bg-paper hover:!bg-line`}
                    onClick={reset}
                  >
                    <Trash2 className="size-4" aria-hidden="true" />
                    Remove
                  </button>
                </div>
              </>
            )}

            {phase.kind === "analyzing" && (
              <>
                <div role="status" aria-live="polite" className="space-y-3">
                  <div className="flex items-center gap-2 text-base font-bold">
                    <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
                    Checking image… {elapsed}s
                  </div>
                  <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-line">
                    <div
                      className="absolute inset-y-0 left-0 w-1/3 rounded-full bg-ink"
                      style={{ animation: "scram-slide 1.4s ease-in-out infinite" }}
                    />
                  </div>
                  <p className="text-sm text-muted">
                    This can take a few seconds. You can cancel at any time.
                  </p>
                </div>
                <button type="button" className={`${secondary} !bg-paper hover:!bg-line`} onClick={cancel}>
                  Cancel
                </button>
              </>
            )}

            {phase.kind === "error" && (
              <>
                <div role="alert" className="flex items-start gap-3 rounded-md border border-ink bg-paper p-4 text-sm">
                  <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
                  <div>
                    <p className="font-bold">Couldn’t analyze this image</p>
                    <p className="mt-1 text-muted">{phase.message}</p>
                  </div>
                </div>
                <button type="button" className={primary} onClick={() => run(phase.file, phase.url)}>
                  <RefreshCw className="size-4" aria-hidden="true" />
                  Try again
                </button>
                <button
                  type="button"
                  className={`${secondary} !bg-paper hover:!bg-line`}
                  onClick={() => inputRef.current?.click()}
                >
                  Choose another image
                </button>
              </>
            )}
          </div>
          {hiddenInput}
        </section>
      )}

      {phase.kind === "done" && (
        <section className="flex flex-col gap-6">
          <ResultView url={phase.url} fileName={phase.file.name} result={phase.result} />
          <div>
            <button type="button" className={secondary} onClick={reset}>
              Analyze another image
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

import { useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";

import Dropzone from "@/components/Dropzone";
import snapshot from "@/data/metrics-snapshot.json";
import { handOff } from "@/lib/analysis";
import type { Metrics } from "@/lib/dashboard";
import { pct } from "@/lib/files";

const metrics = snapshot as Metrics;
const s = metrics.summary;

const STEPS = [
  {
    title: "Keep the original pixels",
    body: "Generators leave faint traces in fine texture and noise. Resizing smooths them away, so the detector looks at full-resolution 224 px patches instead of a shrunken copy.",
  },
  {
    title: "Classify with a vision transformer",
    body: "A ViT-B/16 scores five patches and averages them. It names the most likely source too: a human, or one of the generators it has learned.",
  },
  {
    title: "Map where it looks machine-made",
    body: "Up to 16 patches across the image are scored one by one. That shows where the AI traces are, which matters for human work that was partly edited with AI.",
  },
  {
    title: "Learn new generators without forgetting",
    body: "When a new generator appears, the model learns it from fresh examples while replaying a memory of older ones, so it doesn’t forget what it already knew.",
  },
];

const LIMITS = [
  "It gives a probability, not proof. Don’t use it alone to accuse anyone.",
  "Screenshots, resizing and heavy JPEG compression weaken the traces it looks for.",
  "It is less reliable on generators it hasn’t learned. New ones appear every month.",
  "Human artworks in unusual styles or media can look unfamiliar and be misjudged.",
];

export default function Home() {
  const navigate = useNavigate();
  const onFiles = useCallback(
    (files: File[]) => {
      if (!files[0]) return;
      handOff(files[0]);
      navigate("/analyze");
    },
    [navigate],
  );

  const generators = metrics.experiences.map((e) => e.label);

  return (
    <div className="flex flex-col gap-24 sm:gap-32">
      <section className="grid items-center gap-10 lg:grid-cols-[1.1fr_1fr] lg:gap-16">
        <div>
          <h1 className="text-[clamp(2.75rem,7vw,5.25rem)] font-extrabold leading-[0.95] tracking-[-0.035em]">
            Made by hand, or by a model?
          </h1>
          <p className="mt-6 max-w-xl text-xl leading-relaxed text-muted">
            S.C.R.A.M estimates whether a piece of art is human-made or AI-generated, shows which areas look
            machine-made, and keeps learning as new image generators appear.
          </p>
          <p className="mt-8 text-[15px] text-muted">
            <Link to="/analyze" className="font-semibold text-ink underline decoration-line decoration-2 underline-offset-4 hover:decoration-ink">
              Open the analyzer
            </Link>{" "}
            or{" "}
            <Link to="/dashboard" className="font-semibold text-ink underline decoration-line decoration-2 underline-offset-4 hover:decoration-ink">
              see how the model performs
            </Link>
          </p>
        </div>
        <Dropzone onFiles={onFiles} acceptPaste />
      </section>

      <section aria-labelledby="evidence-title" className="border-t border-line pt-10">
        <h2 id="evidence-title" className="text-2xl font-bold tracking-tight">
          How well the current model does
        </h2>
        <dl className="mt-8 grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <dt className="text-muted">Real vs AI, held-out test images</dt>
            <dd className="tabular mt-1 text-5xl font-extrabold tracking-tight">{pct(s.final_real_vs_ai_accuracy ?? 0)}</dd>
          </div>
          <div>
            <dt className="text-muted">Ranking quality (ROC-AUC)</dt>
            <dd className="tabular mt-1 text-5xl font-extrabold tracking-tight">{(s.final_real_vs_ai_auc ?? 0).toFixed(3)}</dd>
          </div>
          <div>
            <dt className="text-muted">Generators learned one after another</dt>
            <dd className="tabular mt-1 text-5xl font-extrabold tracking-tight">{generators.length}</dd>
          </div>
          <div>
            <dt className="text-muted">Accuracy lost on older generators</dt>
            <dd className="tabular mt-1 text-5xl font-extrabold tracking-tight">
              {((s.final_forgetting ?? 0) * 100).toFixed(1)} pts
            </dd>
          </div>
        </dl>
        <p className="mt-8 max-w-3xl text-muted">
          Learned so far: {generators.join(", ")}. The numbers come from the {metrics.method_label.toLowerCase()} run in
          this project.{" "}
          <Link to="/dashboard" className="font-semibold text-ink underline decoration-line decoration-2 underline-offset-4 hover:decoration-ink">
            Full breakdown
          </Link>
        </p>
      </section>

      <section id="how" aria-labelledby="how-title" className="scroll-mt-24">
        <h2 id="how-title" className="text-4xl font-extrabold tracking-tight sm:text-5xl">
          How it works
        </h2>
        <ol className="mt-10 grid gap-x-12 gap-y-10 md:grid-cols-2">
          {STEPS.map((step, i) => (
            <li key={step.title} className="grid grid-cols-[auto_1fr] gap-x-5">
              <span className="tabular flex size-10 items-center justify-center rounded-full border border-line font-bold text-muted">
                {i + 1}
              </span>
              <div>
                <h3 className="text-xl font-bold tracking-tight">{step.title}</h3>
                <p className="mt-2 max-w-prose text-muted">{step.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="grid gap-12 md:grid-cols-2">
        <div id="limits" className="scroll-mt-24">
          <h2 className="text-3xl font-extrabold tracking-tight">What it can’t tell you</h2>
          <ul className="mt-6 space-y-3">
            {LIMITS.map((l) => (
              <li key={l} className="flex gap-3">
                <span aria-hidden="true" className="mt-[0.6em] h-px w-4 shrink-0 bg-muted" />
                {l}
              </li>
            ))}
          </ul>
        </div>
        <div id="privacy" className="scroll-mt-24">
          <h2 className="text-3xl font-extrabold tracking-tight">What happens to your image</h2>
          <p className="mt-6 max-w-prose">
            Your image is sent to the detector server, analyzed, and the result is sent back. The image itself isn’t kept
            unless whoever runs the server turns that on. If history is enabled, only a fingerprint of the file and the
            result are saved.
          </p>
        </div>
      </section>
    </div>
  );
}

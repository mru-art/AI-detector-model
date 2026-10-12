import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

const POINTS = [
  {
    title: "Classify",
    body: "Checks whether an image is likely real or AI-generated.",
  },
  {
    title: "Flag",
    body: "Marks the regions that influenced the result.",
  },
  {
    title: "Keep learning",
    body: "Adapts as new image generators appear, without forgetting older ones.",
  },
];

const STEPS = [
  {
    title: "Look at more than colors",
    body: "The detector compares ordinary image data with frequency-domain and residual/noise cues, looking for traces that generation leaves behind.",
  },
  {
    title: "Classify with a vision transformer",
    body: "A vision transformer (ViT) splits the image into patches and outputs a real-vs-AI decision with a confidence value.",
  },
  {
    title: "Flag regions",
    body: "Attention-based explainability (Attention Rollout or Grad-CAM) turns the model’s focus into a mask of regions that influenced the result.",
  },
  {
    title: "Keep learning",
    body: "As new generators appear, continual learning (experience replay, EWC and knowledge distillation) updates the detector while limiting forgetting of older ones.",
  },
];

const LIMITS = [
  "Results are probabilities, not proof.",
  "Detectors can latch onto cues specific to one generator, dataset or kind of scene, and lose accuracy on generators they haven’t seen.",
  "Resizing and JPEG compression can weaken the signals the detector relies on.",
  "New generators keep appearing, so a result that was reliable last month may be less reliable today.",
];

export default function Home() {
  return (
    <div className="flex flex-col gap-24">
      <section className="flex min-h-[62svh] flex-col justify-center">
        <div className="@container">
          <h1 className="whitespace-nowrap text-center font-display text-[length:29.5cqw] leading-[0.9] tracking-tight">
            S.C.R.A.M
          </h1>
        </div>
        <p className="mt-6 text-center text-sm font-bold uppercase tracking-widest text-muted sm:text-base">
          Spectral Classification, Replay and Art Masking
        </p>
      </section>

      <section className="grid gap-8 border-t border-line pt-10 md:grid-cols-[1.2fr_2fr]">
        <div>
          <p className="text-2xl font-bold leading-snug sm:text-3xl">
            A research prototype that estimates whether an image is real or AI-generated.
          </p>
          <Link
            to="/analyze"
            className="mt-8 inline-flex items-center gap-2 rounded-md bg-ink px-6 py-3.5 text-base font-bold text-paper transition-opacity hover:opacity-85"
          >
            Analyze an image
            <ArrowRight className="size-4" aria-hidden="true" />
          </Link>
        </div>
        <ul className="grid gap-4 sm:grid-cols-3">
          {POINTS.map((p) => (
            <li key={p.title} className="rounded-md bg-mist p-5">
              <h2 className="text-lg font-bold">{p.title}</h2>
              <p className="mt-2 text-sm leading-relaxed text-muted">{p.body}</p>
            </li>
          ))}
        </ul>
      </section>

      <section id="how" aria-labelledby="how-title" className="scroll-mt-28">
        <h2 id="how-title" className="font-display text-5xl uppercase leading-none sm:text-7xl">
          How it’s designed to work
        </h2>
        <p className="mt-4 max-w-2xl text-base text-muted">
          This describes the planned design of the prototype and may change as it is built.
        </p>
        <ol className="mt-8 grid gap-4 md:grid-cols-2">
          {STEPS.map((s, i) => (
            <li key={s.title} className="rounded-md bg-mist p-6">
              <p className="font-display text-5xl leading-none text-muted">
                {String(i + 1).padStart(2, "0")}
              </p>
              <h3 className="mt-4 text-xl font-bold">{s.title}</h3>
              <p className="mt-2 text-base leading-relaxed text-muted">{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section id="limits" aria-labelledby="limits-title" className="scroll-mt-28">
        <h2 id="limits-title" className="font-display text-5xl uppercase leading-none sm:text-7xl">
          Limitations
        </h2>
        <ul className="mt-8 grid max-w-3xl gap-3">
          {LIMITS.map((l) => (
            <li key={l} className="flex gap-3 text-base leading-relaxed">
              <span aria-hidden="true" className="mt-2.5 size-1.5 shrink-0 rounded-full bg-ink" />
              {l}
            </li>
          ))}
        </ul>
      </section>

      <section id="privacy" aria-labelledby="privacy-title" className="scroll-mt-28">
        <h2 id="privacy-title" className="font-display text-5xl uppercase leading-none sm:text-7xl">
          Privacy
        </h2>
        <div className="mt-8 max-w-2xl rounded-md bg-mist p-6 text-base leading-relaxed">
          <p>
            In this demo, the image you choose stays in your browser and the results are generated
            locally as a mock. Nothing is uploaded. When the detector is connected, this section
            will explain what happens to uploaded images.
          </p>
        </div>
      </section>

      <p className="text-sm text-muted">
        Results are estimates, not proof.{" "}
        <a href="#limits" className="font-bold text-ink underline underline-offset-4">
          See the limitations
        </a>
      </p>
    </div>
  );
}

import PageHeader from "@/components/PageHeader";

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

export default function Method() {
  return (
    <div className="flex flex-col gap-14">
      <PageHeader title="Method" draft>
        How S.C.R.A.M is designed to work, and where it falls short. Wording follows the project
        proposal and should be checked against what the team has actually built.
      </PageHeader>

      <section aria-labelledby="how">
        <h2 id="how" className="sr-only">
          How it works
        </h2>
        <ol className="grid gap-4 md:grid-cols-2">
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

      <section aria-labelledby="limits" className="border-t border-line pt-10">
        <h2 id="limits" className="font-display text-4xl uppercase sm:text-5xl">
          Limitations
        </h2>
        <ul className="mt-6 grid max-w-3xl gap-3">
          {LIMITS.map((l) => (
            <li key={l} className="flex gap-3 text-base leading-relaxed">
              <span aria-hidden="true" className="mt-2.5 size-1.5 shrink-0 rounded-full bg-ink" />
              {l}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

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

export default function Home() {
  return (
    <div className="flex flex-col gap-16">
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

      <p className="text-sm text-muted">
        Results are estimates, not proof.{" "}
        <Link to="/method" className="font-bold text-ink underline underline-offset-4">
          How it works and where it falls short
        </Link>
      </p>
    </div>
  );
}
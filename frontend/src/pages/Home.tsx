import { useLayoutEffect, useRef } from "react";
import type { MouseEvent } from "react";
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

/**
 * The big wordmark shrinks as you scroll and settles into the top bar,
 * where it stays as the site logo. It is a fixed element whose font size and
 * position follow the scroll until it docks. After that nothing changes, so it
 * stays perfectly still. Real font sizes (no scaling) keep the text sharp.
 * The header's own logo is hidden on this page meanwhile (index.css).
 */
function Hero() {
  const heroRef = useRef<HTMLElement>(null);
  const slotRef = useRef<HTMLDivElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);

  useLayoutEffect(() => {
    const hero = heroRef.current;
    const slot = slotRef.current;
    const heading = headingRef.current;
    if (!hero || !slot || !heading) return;

    const root = document.documentElement;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let big = 100;
    let small = 30;
    let slotTop = 0;
    let targetTop = 20;
    let distance = 1;
    let frame = 0;
    let lastSize = NaN;
    let lastTop = NaN;
    let lastProgress = NaN;

    const update = () => {
      frame = 0;
      const y = Math.max(0, window.scrollY);
      if (reduce) {
        // No movement: just swap in the header logo once the hero is scrolled past.
        if (y > distance * 0.6) root.removeAttribute("data-logo");
        else root.setAttribute("data-logo", "hidden");
        return;
      }
      const progress = Math.min(1, y / distance);
      const eased = progress * progress * (3 - 2 * progress); // gentle start and landing
      const size = Math.round(big * Math.pow(small / big, eased) * 100) / 100;
      const top = Math.round(Math.max(targetTop, slotTop - y) * 100) / 100;
      // Only touch the page when something actually changed (nothing does once docked).
      if (size !== lastSize) {
        heading.style.fontSize = `${size}px`;
        lastSize = size;
      }
      if (top !== lastTop) {
        heading.style.top = `${top}px`;
        lastTop = top;
      }
      if (progress !== lastProgress) {
        hero.style.setProperty("--p", String(progress));
        lastProgress = progress;
      }
    };

    const measure = () => {
      const header = document.querySelector("header");
      const headerHeight = header ? header.getBoundingClientRect().height : 68;
      small = window.innerWidth >= 640 ? 30 : 24; // header logo size in px
      big = Math.max(small, slot.getBoundingClientRect().width * 0.295);
      if (reduce) {
        heading.style.position = "static";
        heading.style.fontSize = `${big}px`;
        slot.style.height = "";
      } else {
        slot.style.height = `${big * 0.9}px`; // keeps the page layout steady
      }
      slotTop = slot.getBoundingClientRect().top + window.scrollY;
      targetTop = headerHeight / 2 - 0.45 * small;
      distance = Math.max(1, slotTop - targetTop);
      lastSize = NaN;
      lastTop = NaN;
      lastProgress = NaN;
      update();
    };

    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };

    root.setAttribute("data-logo", "hidden");
    measure();
    void document.fonts?.ready.then(measure);
    const observer = new ResizeObserver(measure);
    observer.observe(slot);
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", measure);

    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", measure);
      observer.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
      root.removeAttribute("data-logo");
    };
  }, []);

  const toTop = (event: MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
  };

  return (
    <section ref={heroRef} className="flex min-h-[62svh] flex-col justify-center">
      <div ref={slotRef}>
        <h1
          ref={headingRef}
          className="pointer-events-none fixed inset-x-0 z-[60] whitespace-nowrap text-center font-display leading-[0.9] tracking-tight"
          style={{ top: "6rem", fontSize: "20vw" }}
        >
          <Link to="/" onClick={toTop} className="pointer-events-auto inline-block">
            S.C.R.A.M
          </Link>
        </h1>
      </div>
      <p
        className="mt-6 text-center text-sm font-bold uppercase tracking-widest text-muted sm:text-base"
        style={{ opacity: "calc(1 - var(--p, 0) * 1.6)" }}
      >
        Spectral Classification, Replay and Art Masking
      </p>
    </section>
  );
}

export default function Home() {
  return (
    <div className="flex flex-col gap-24">
      <Hero />

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

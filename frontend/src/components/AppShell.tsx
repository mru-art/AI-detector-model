import { useEffect } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";

import ThemeToggle from "@/components/ThemeToggle";
import { useHealth, type HealthState } from "@/lib/api";

const LINKS = [
  { to: "/analyze", label: "Analyze" },
  { to: "/dashboard", label: "Model results", short: "Results" },
  { to: "/#how", label: "How it works" },
];

function statusOf(s: HealthState): { color: string; label: string; detail: string } {
  switch (s.kind) {
    case "ready":
      return { color: "bg-human", label: "Detector ready", detail: s.health.model.version ?? "" };
    case "loading":
      return { color: "bg-warn animate-pulse", label: "Loading model", detail: "" };
    case "no-model":
      return { color: "bg-ai", label: "No model loaded", detail: s.health.model.error ?? "" };
    case "offline":
      return { color: "bg-ai", label: "Detector offline", detail: s.message };
    case "mock":
      return { color: "bg-warn", label: "Demo mode", detail: "Results are made up" };
    default:
      return { color: "bg-line animate-pulse", label: "Checking detector", detail: "" };
  }
}

function StatusDot() {
  const s = statusOf(useHealth());
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted" title={s.detail || s.label}>
      <span className={`size-2 rounded-full ${s.color}`} aria-hidden="true" />
      <span className="hidden md:inline">{s.label}</span>
      <span className="sr-only md:hidden">{s.label}</span>
    </span>
  );
}

export default function AppShell() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) {
      document.getElementById(hash.slice(1))?.scrollIntoView();
    } else {
      window.scrollTo(0, 0);
    }
  }, [pathname, hash]);

  return (
    <div className="flex min-h-svh flex-col">
      <a
        href="#main"
        className="sr-only z-50 rounded bg-ink px-4 py-2 text-paper focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-40 border-b border-line bg-paper/85 backdrop-blur-md">
        <div className="mx-auto flex h-16 w-full max-w-6xl items-center gap-4 px-4 sm:px-6">
          <Link to="/" className="text-xl font-extrabold tracking-tight" aria-label="S.C.R.A.M home">
            S.C.R.A.M
          </Link>
          <nav aria-label="Main" className="ml-auto flex items-center gap-1 sm:ml-8 sm:mr-auto">
            {LINKS.map((l) =>
              l.to.includes("#") ? (
                <Link
                  key={l.to}
                  to={l.to}
                  className="hidden rounded-full px-3 py-1.5 text-[15px] text-muted transition-colors hover:text-ink sm:inline"
                >
                  {l.label}
                </Link>
              ) : (
                <NavLink
                  key={l.to}
                  to={l.to}
                  className={({ isActive }) =>
                    `rounded-full px-3 py-1.5 text-[15px] transition-colors ${
                      isActive ? "bg-sunken font-semibold text-ink" : "text-muted hover:text-ink"
                    }`
                  }
                >
                  {"short" in l ? (
                    <>
                      <span className="sm:hidden">{l.short}</span>
                      <span className="hidden sm:inline">{l.label}</span>
                    </>
                  ) : (
                    l.label
                  )}
                </NavLink>
              ),
            )}
          </nav>
          <div className="flex items-center gap-2">
            <StatusDot />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 pb-24 pt-10 sm:px-6 sm:pt-14">
        <Outlet />
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap justify-between gap-x-8 gap-y-2 px-4 py-6 text-sm text-muted sm:px-6">
          <p>S.C.R.A.M: Spectral Classification, Replay and Art Masking. A research prototype.</p>
          <p>Results are estimates, not proof.</p>
        </div>
      </footer>
    </div>
  );
}

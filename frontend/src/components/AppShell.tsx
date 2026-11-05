import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { Menu as MenuIcon, X } from "lucide-react";

import ThemeToggle from "@/components/ThemeToggle";

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/analyze", label: "Analyze" },
  { to: "/dashboard", label: "Dashboard" },
];

function MenuButton() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { pathname } = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative justify-self-start">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="site-menu"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-md bg-mist px-4 text-base font-bold transition-colors hover:bg-line"
      >
        {open ? (
          <X className="size-4" aria-hidden="true" />
        ) : (
          <MenuIcon className="size-4" aria-hidden="true" />
        )}
        {open ? "Close" : "Menu"}
      </button>
      {open && (
        <nav
          id="site-menu"
          aria-label="Main"
          className="absolute left-0 top-full mt-2 flex w-56 flex-col rounded-md bg-mist p-1.5"
        >
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) =>
                `rounded px-4 py-2.5 text-base font-bold transition-colors hover:bg-line ${
                  isActive ? "text-ink" : "text-muted hover:text-ink"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      )}
    </div>
  );
}

export default function AppShell() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="min-h-svh">
      <header className="fixed inset-x-0 top-0 z-50 bg-paper/85 pt-[env(safe-area-inset-top)] backdrop-blur">
        <div className="mx-auto grid w-full max-w-[1400px] grid-cols-[1fr_auto_1fr] items-center px-5 py-3 sm:px-8">
          <MenuButton />
          <Link
            to="/"
            aria-label="S.C.R.A.M home"
            data-site-logo
            className="justify-self-center font-display text-2xl leading-none tracking-tight sm:text-3xl"
          >
            S.C.R.A.M
          </Link>
          <div className="justify-self-end">
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1400px] px-5 pb-16 pt-24 sm:px-8">
        <Outlet />
      </main>
    </div>
  );
}

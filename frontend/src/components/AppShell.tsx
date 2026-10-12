import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import ThemeToggle from "@/components/ThemeToggle";

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/analyze", label: "Analyze" },
  { to: "/dashboard", label: "Dashboard" },
];

function MenuPill() {
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
    <div
      ref={ref}
      className="fixed left-1/2 top-[max(1rem,env(safe-area-inset-top))] z-50 w-56 -translate-x-1/2"
    >
      <button
        type="button"
        aria-expanded={open}
        aria-controls="site-menu"
        onClick={() => setOpen((v) => !v)}
        className="w-full cursor-pointer rounded-md bg-mist py-3 text-center text-base font-bold transition-colors hover:bg-line"
      >
        {open ? "Close" : "Menu"}
      </button>
      {open && (
        <nav
          id="site-menu"
          aria-label="Main"
          className="mt-2 flex flex-col rounded-md bg-mist p-1.5"
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
      <MenuPill />
      <ThemeToggle />
      <main className="mx-auto w-full max-w-[1400px] px-5 pb-24 pt-28 sm:px-8">
        <Outlet />
      </main>
    </div>
  );
}

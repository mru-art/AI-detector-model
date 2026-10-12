import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="flex min-h-[55svh] flex-col items-start justify-center gap-6">
      <p className="font-display text-[clamp(6rem,24vw,18rem)] leading-[0.85]">404</p>
      <p className="text-xl font-bold">That page doesn’t exist.</p>
      <Link
        to="/"
        className="rounded-md bg-ink px-6 py-3.5 text-base font-bold text-paper transition-opacity hover:opacity-85"
      >
        Back to home
      </Link>
    </div>
  );
}

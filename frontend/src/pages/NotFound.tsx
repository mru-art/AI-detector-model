import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="flex min-h-[50svh] flex-col items-start justify-center gap-5">
      <h1 className="text-5xl font-extrabold tracking-tight">This page doesn’t exist</h1>
      <p className="text-lg text-muted">The link may be old, or the address mistyped.</p>
      <Link to="/" className="rounded-full bg-ink px-5 py-3 font-semibold text-paper transition-opacity hover:opacity-85">
        Go to the home page
      </Link>
    </div>
  );
}

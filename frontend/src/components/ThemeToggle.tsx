import { Moon, Sun } from "lucide-react";

import { useTheme } from "@/lib/theme";

export default function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const dark = theme === "dark";

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      className="inline-flex min-h-11 min-w-11 cursor-pointer items-center justify-center gap-2 rounded-md bg-mist px-4 text-base font-bold transition-colors hover:bg-line"
    >
      {dark ? (
        <Sun className="size-4" aria-hidden="true" />
      ) : (
        <Moon className="size-4" aria-hidden="true" />
      )}
      <span className="hidden sm:inline">{dark ? "Light" : "Dark"}</span>
    </button>
  );
}
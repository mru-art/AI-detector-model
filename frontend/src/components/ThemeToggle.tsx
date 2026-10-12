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
      className="fixed bottom-[max(1rem,env(safe-area-inset-bottom))] left-1/2 z-40 inline-flex -translate-x-1/2 cursor-pointer items-center gap-2 rounded-md bg-mist px-5 py-3 text-base font-bold transition-colors hover:bg-line"
    >
      {dark ? (
        <Sun className="size-4" aria-hidden="true" />
      ) : (
        <Moon className="size-4" aria-hidden="true" />
      )}
      {dark ? "Light" : "Dark"}
    </button>
  );
}

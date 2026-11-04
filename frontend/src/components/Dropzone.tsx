import { useEffect, useRef, useState } from "react";
import { ImagePlus } from "lucide-react";

import { ALLOWED_LABEL, ALLOWED_TYPES, MAX_MB } from "@/lib/files";

type Props = {
  onFiles: (files: File[]) => void;
  /** Also accept images pasted with Ctrl/Cmd+V anywhere on the page. */
  acceptPaste?: boolean;
  size?: "large" | "compact";
};

export default function Dropzone({ onFiles, acceptPaste = true, size = "large" }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);

  useEffect(() => {
    if (!acceptPaste) return;
    const onPaste = (e: ClipboardEvent) => {
      const files = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith("image/"));
      if (files.length) {
        e.preventDefault();
        onFiles(files);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [acceptPaste, onFiles]);

  const large = size === "large";

  return (
    <div
      onDragEnter={(e) => {
        e.preventDefault();
        depth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => {
        depth.current -= 1;
        if (depth.current <= 0) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        depth.current = 0;
        setDragging(false);
        onFiles(Array.from(e.dataTransfer.files));
      }}
      className={`relative flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
        large ? "min-h-[340px]" : "min-h-[220px]"
      } ${dragging ? "border-human bg-human/8" : "border-line bg-surface"}`}
    >
      <span
        className={`flex size-14 items-center justify-center rounded-full transition-colors ${
          dragging ? "bg-human text-white" : "bg-sunken text-ink"
        }`}
      >
        <ImagePlus className="size-6" aria-hidden="true" />
      </span>
      <div>
        <p className={`font-bold tracking-tight ${large ? "text-2xl sm:text-3xl" : "text-xl"}`}>
          {dragging ? "Drop it to analyze" : "Drop an artwork here"}
        </p>
        <p className="mt-1 text-muted">
          or{" "}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="cursor-pointer font-semibold text-human underline decoration-2 underline-offset-4 hover:decoration-4"
          >
            choose a file
          </button>
          {acceptPaste && <span className="hidden sm:inline">, or paste one</span>}
        </p>
      </div>
      <p className="text-sm text-muted">
        {ALLOWED_LABEL}, up to {MAX_MB} MB
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_TYPES.join(",")}
        className="sr-only"
        tabIndex={-1}
        aria-label="Choose an image"
        onChange={(e) => {
          onFiles(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
    </div>
  );
}

export const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;
export const ALLOWED_LABEL = "JPG, PNG or WebP";
/** Keep in step with MAX_UPLOAD_MB in backend/api/detections.py. */
export const MAX_MB = 15;

export function validateImage(file: File): string | null {
  if (!(ALLOWED_TYPES as readonly string[]).includes(file.type)) {
    return `“${file.name}” isn’t a supported type. Use ${ALLOWED_LABEL}.`;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    return `“${file.name}” is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is ${MAX_MB} MB.`;
  }
  if (file.size === 0) return `“${file.name}” is empty.`;
  return null;
}

export function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export const pct = (n: number, digits = 0): string => `${(n * 100).toFixed(digits)}%`;

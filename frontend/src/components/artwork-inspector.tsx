import { useCallback, useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import {
  AlertCircle,
  ArrowLeftRight,
  ImageIcon,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

type Artwork = {
  id: string;
  artist_id: string;
  original_uri: string;
  protected_uri: string | null;
  original_download_url: string;
  protected_download_url: string | null;
  created_at: string;
  sha256: string;
};

type Detection = {
  id: string;
  artwork_id: string;
  model_version: string;
  prediction: string;
  ai_probability: number;
  confidence: number;
  generator: string;
  timestamp: string;
};

type ArtworkInspectorProps = {
  artworkId: string;
};

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

function confidenceLabel(value: number): "High" | "Medium" | "Low" {
  if (value >= 0.85) return "High";
  if (value >= 0.6) return "Medium";
  return "Low";
}

function confidenceVariant(
  value: number,
): "default" | "secondary" | "destructive" | "outline" {
  if (value >= 0.85) return "default";
  if (value >= 0.6) return "secondary";
  return "outline";
}

function probabilityColor(probability: number): string {
  if (probability >= 0.7) return "bg-red-500";
  if (probability >= 0.4) return "bg-amber-500";
  return "bg-emerald-500";
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

async function readError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (
      typeof body === "object" &&
      body !== null &&
      "detail" in body &&
      typeof body.detail === "string"
    ) {
      return body.detail;
    }
  } catch {
    // Fall back to the HTTP status message.
  }
  return `Request failed (${response.status})`;
}

export default function ArtworkInspector({
  artworkId,
}: ArtworkInspectorProps) {
  const [artwork, setArtwork] = useState<Artwork | null>(null);
  const [detection, setDetection] = useState<Detection | null>(null);
  const [loading, setLoading] = useState(true);
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparePosition, setComparePosition] = useState(50);

  const runDetection = useCallback(
    async (signal?: AbortSignal) => {
      setDetecting(true);
      try {
        const form = new FormData();
        form.set("artwork_id", artworkId);

        const response = await fetch(`${API_BASE_URL}/detect`, {
          method: "POST",
          body: form,
          signal,
        });

        if (!response.ok) {
          throw new Error(await readError(response));
        }

        setDetection((await response.json()) as Detection);
      } finally {
        setDetecting(false);
      }
    },
    [artworkId],
  );

  useEffect(() => {
    const controller = new AbortController();

    async function loadInspector(): Promise<void> {
      setLoading(true);
      setError(null);
      setArtwork(null);
      setDetection(null);

      try {
        const artworkResponse = await fetch(
          `${API_BASE_URL}/artworks/${encodeURIComponent(artworkId)}`,
          { signal: controller.signal },
        );

        if (!artworkResponse.ok) {
          throw new Error(await readError(artworkResponse));
        }

        const artworkData = (await artworkResponse.json()) as Artwork;
        setArtwork(artworkData);
        setLoading(false);

        await runDetection(controller.signal);
      } catch (caught) {
        if (controller.signal.aborted) return;
        setError(
          caught instanceof Error ? caught.message : "Unable to load artwork",
        );
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    void loadInspector();
    return () => controller.abort();
  }, [artworkId, runDetection]);

  const handleCompareChange = (event: ChangeEvent<HTMLInputElement>) => {
    setComparePosition(Number(event.target.value));
  };

  if (loading && !artwork) {
    return (
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.7fr)_minmax(280px,0.8fr)]">
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-72" />
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <Skeleton className="aspect-[4/3] w-full rounded-lg" />
            <Skeleton className="aspect-[4/3] w-full rounded-lg" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-4 w-56" />
          </CardHeader>
          <CardContent className="space-y-5">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-8 w-32" />
            <Skeleton className="h-8 w-28" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error && !artwork) {
    return (
      <Card className="border-destructive/40">
        <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
          <AlertCircle className="size-8 text-destructive" />
          <div>
            <p className="font-medium">Couldn’t load this artwork</p>
            <p className="mt-1 text-sm text-muted-foreground">{error}</p>
          </div>
          <Button
            variant="outline"
            onClick={() => window.location.reload()}
            className="mt-2"
          >
            <RefreshCw className="mr-2 size-4" />
            Try again
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!artwork) return null;

  const aiProbability = detection
    ? Math.min(1, Math.max(0, detection.ai_probability))
    : null;
  const confidence = detection
    ? Math.min(1, Math.max(0, detection.confidence))
    : null;
  const isAi = detection?.prediction.toUpperCase() === "AI";

  return (
    <div className="grid min-w-0 gap-6 lg:grid-cols-[minmax(0,1.7fr)_minmax(280px,0.8fr)]">
      <Card className="min-w-0 overflow-hidden">
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle>Artwork inspector</CardTitle>
              <CardDescription className="mt-1 break-all">
                Artwork ID: {artwork.id}
              </CardDescription>
            </div>
            <Badge variant="outline" className="shrink-0">
              <ShieldCheck className="mr-1 size-3.5" />
              Protected view
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-5">
          <div className="grid gap-4 md:grid-cols-2">
            <ImagePanel
              title="Original artwork"
              imageUrl={artwork.original_download_url}
              alt="Original artwork"
            />
            <ImagePanel
              title="Protected representation"
              imageUrl={artwork.protected_download_url}
              alt="Protected artwork representation"
            />
          </div>

          {artwork.protected_download_url && (
            <div className="space-y-3 rounded-lg border p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">Visual comparison</p>
                  <p className="text-xs text-muted-foreground">
                    Drag the slider to reveal the protected image.
                  </p>
                </div>
                <ArrowLeftRight className="size-4 shrink-0 text-muted-foreground" />
              </div>

              <div className="relative mx-auto aspect-[4/3] max-h-[520px] overflow-hidden rounded-md bg-muted">
                <img
                  src={artwork.original_download_url}
                  alt="Original artwork comparison"
                  className="absolute inset-0 size-full object-contain"
                />
                <div
                  className="absolute inset-0 overflow-hidden"
                  style={{ clipPath: `inset(0 ${100 - comparePosition}% 0 0)` }}
                >
                  <img
                    src={artwork.protected_download_url}
                    alt="Protected artwork comparison overlay"
                    className="absolute inset-0 size-full object-contain"
                  />
                </div>
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-y-0 z-10 w-0.5 bg-white shadow"
                  style={{ left: `${comparePosition}%` }}
                >
                  <span className="absolute left-1/2 top-1/2 flex size-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border bg-background shadow">
                    <ArrowLeftRight className="size-4" />
                  </span>
                </div>
                <span className="absolute left-3 top-3 rounded bg-background/90 px-2 py-1 text-xs font-medium">
                  Protected
                </span>
                <span className="absolute right-3 top-3 rounded bg-background/90 px-2 py-1 text-xs font-medium">
                  Original
                </span>
              </div>

              <input
                aria-label="Compare original and protected artwork"
                type="range"
                min={0}
                max={100}
                value={comparePosition}
                onChange={handleCompareChange}
                className="w-full accent-primary"
              />
            </div>
          )}

          <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
            <span>Artist: {artwork.artist_id}</span>
            <span>
              Added: {new Date(artwork.created_at).toLocaleDateString()}
            </span>
          </div>
        </CardContent>
      </Card>

      <Card className="h-fit">
        <CardHeader>
          <CardTitle>Detection summary</CardTitle>
          <CardDescription>
            Forensic analysis for this artwork.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {detecting && !detection ? (
            <div className="space-y-5" aria-label="Running detection">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <LoaderCircle className="size-4 animate-spin" />
                Running detector…
              </div>
              <Skeleton className="h-10 w-36" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-8 w-28" />
              <Skeleton className="h-8 w-32" />
            </div>
          ) : detection && aiProbability !== null && confidence !== null ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <span className="text-sm font-medium">Classification</span>
                <Badge variant={isAi ? "destructive" : "default"}>
                  {isAi ? "AI Generated" : "Human Artwork"}
                </Badge>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">AI probability</span>
                  <span className="font-semibold">
                    {formatPercent(aiProbability)}
                  </span>
                </div>
                <div
                  className="h-2.5 overflow-hidden rounded-full bg-muted"
                  role="progressbar"
                  aria-label="AI probability"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={aiProbability * 100}
                >
                  <div
                    className={`h-full rounded-full transition-[width] duration-500 ${probabilityColor(aiProbability)}`}
                    style={{ width: `${aiProbability * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Human</span>
                  <span>AI</span>
                </div>
              </div>

              <div className="flex items-center justify-between gap-3">
                <span className="text-sm text-muted-foreground">Confidence</span>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">
                    {formatPercent(confidence)}
                  </span>
                  <Badge variant={confidenceVariant(confidence)}>
                    {confidenceLabel(confidence)}
                  </Badge>
                </div>
              </div>

              <InfoRow label="Generator" value={detection.generator} />
              <InfoRow label="Detector version" value={detection.model_version} />

              <p className="text-xs text-muted-foreground">
                Analyzed{" "}
                {new Date(detection.timestamp).toLocaleString()}
              </p>
            </>
          ) : (
            <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
              Detection results are not available.
              {error && <p className="mt-1 text-destructive">{error}</p>}
            </div>
          )}

          <Button
            variant="outline"
            className="w-full"
            disabled={detecting}
            onClick={() => {
              setError(null);
              void runDetection().catch((caught: unknown) => {
                setError(
                  caught instanceof Error
                    ? caught.message
                    : "Detection request failed",
                );
              });
            }}
          >
            {detecting ? (
              <LoaderCircle className="mr-2 size-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 size-4" />
            )}
            {detection ? "Run detection again" : "Retry detection"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function ImagePanel({
  title,
  imageUrl,
  alt,
}: {
  title: string;
  imageUrl: string | null;
  alt: string;
}) {
  return (
    <div className="min-w-0 space-y-2">
      <h3 className="text-sm font-medium">{title}</h3>
      <div className="flex aspect-[4/3] items-center justify-center overflow-hidden rounded-lg border bg-muted">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={alt}
            className="size-full object-contain"
            loading="lazy"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 px-4 text-center text-sm text-muted-foreground">
            <ImageIcon className="size-7" />
            <span>No protected image is available</span>
          </div>
        )}
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-muted-foreground">{label}</span>
      <Badge variant="outline" className="max-w-[60%] truncate">
        {value}
      </Badge>
    </div>
  );
}
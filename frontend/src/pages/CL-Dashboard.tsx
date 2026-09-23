import {
  useCallback,
  useEffect,
  useState,
  type ReactNode,
  type SVGProps,
} from "react";

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps & { children?: ReactNode }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {children}
    </svg>
  );
}

const Activity = (props: IconProps) => <Icon {...props}><path d="M3 12h4l3-8 4 16 3-8h4" /></Icon>;
const AlertCircle = (props: IconProps) => <Icon {...props}><circle cx="12" cy="12" r="9" /><path d="M12 8v4m0 4h.01" /></Icon>;
const CheckCircle2 = (props: IconProps) => <Icon {...props}><circle cx="12" cy="12" r="9" /><path d="m8 12 2.5 2.5L16 9" /></Icon>;
const Clock3 = (props: IconProps) => <Icon {...props}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></Icon>;
const FlaskConical = (props: IconProps) => <Icon {...props}><path d="M9 3h6m-5 0v6l-5 8a2 2 0 0 0 1.7 3h10.6A2 2 0 0 0 19 17l-5-8V3M7 16h10" /></Icon>;
const Gauge = (props: IconProps) => <Icon {...props}><path d="M4 15a8 8 0 1 1 16 0" /><path d="m12 13 3-3M6 18h12" /></Icon>;
const RefreshCw = (props: IconProps) => <Icon {...props}><path d="M20 11a8 8 0 0 0-14-5L4 8m0-5v5h5M4 13a8 8 0 0 0 14 5l2-2m0 5v-5h-5" /></Icon>;
const Shield = (props: IconProps) => <Icon {...props}><path d="M12 3 20 6v5c0 5-3.5 8-8 10-4.5-2-8-5-8-10V6l8-3Z" /></Icon>;
const Sparkles = (props: IconProps) => <Icon {...props}><path d="m12 3-1.2 4.8L6 9l4.8 1.2L12 15l1.2-4.8L18 9l-4.8-1.2L12 3ZM19 15l-.6 2.4L16 18l2.4.6L19 21l.6-2.4L22 18l-2.4-.6L19 15Z" /></Icon>;
const TrendingUp = (props: IconProps) => <Icon {...props}><path d="m3 17 6-6 4 4 8-8M15 7h6v6" /></Icon>;
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

type HealthResponse = {
  checkpoint: { version: string; parent_version: string | null } | null;
  metrics: {
    stability: number;
    plasticity: number;
    generalization: number;
    forgetting_rate: number;
    calibration_ece: number;
  };
  catastrophic_forgetting_status: "LOW" | "MEDIUM" | "HIGH";
};

type HistoryResponse = {
  updates: Array<{
    version: string;
    accuracy: number;
    stability: number;
    generalization?: number | null;
  }>;
};

type EvaluationRun = {
  run_id: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
  active_version: string;
  candidate_checkpoint_uri: string;
  evaluation_dataset_version: string;
  created_at: string;
  completed_at: string | null;
  breakdown: {
    generators: Record<
      string,
      { accuracy: number | null; sample_count: number }
    >;
    attacker_resistance_scores: Record<string, number>;
  } | null;
  error: string | null;
};

const GENERATORS = [
  "SDXL",
  "FLUX",
  "DALL-E",
  "Midjourney",
  "Firefly",
  "Unknown",
] as const;

const ATTACKER_LABELS: Record<string, string> = {
  AttackerA_HTTPDownloader: "A · Simple HTTP",
  AttackerB_ScreenshotScraper: "B · Screenshot scraper",
  AttackerC_PreprocessingScraper: "C · Preprocessing",
  AttackerD_DenoisingScraper: "D · Denoising",
  AttackerE_MLReconstructionScraper: "E · ML reconstruction",
};

function percent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "—";
}

function scorePercent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(1)}%`
    : "—";
}

async function responseError(response: Response): Promise<string> {
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
    // Use the status message below.
  }
  return `Request failed (${response.status})`;
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(await responseError(response));
  return (await response.json()) as T;
}

function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  accent,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: typeof Activity;
  accent: string;
}) {
  return (
    <Card className="border-slate-800 bg-slate-900/70">
      <CardContent className="flex items-start justify-between gap-4 p-5">
        <div className="min-w-0">
          <p className="text-sm text-slate-400">{title}</p>
          <p className="mt-2 text-3xl font-semibold tracking-tight text-white">
            {value}
          </p>
          <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
        </div>
        <div className={`rounded-lg border border-slate-700 bg-slate-800 p-2.5 ${accent}`}>
          <Icon className="size-5" />
        </div>
      </CardContent>
    </Card>
  );
}

function LoadingDashboard() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, index) => (
          <Skeleton key={index} className="h-32 rounded-xl bg-slate-800" />
        ))}
      </div>
      <div className="grid gap-6 xl:grid-cols-2">
        <Skeleton className="h-80 rounded-xl bg-slate-800" />
        <Skeleton className="h-80 rounded-xl bg-slate-800" />
      </div>
    </div>
  );
}

export default function CLDashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse["updates"]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationRun | null>(null);
  const [candidateUri, setCandidateUri] = useState("");
  const [datasetVersion, setDatasetVersion] = useState("latest");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const loadDashboard = useCallback(async (signal?: AbortSignal) => {
    setRefreshing(true);
    setError(null);
    try {
      const [healthResult, historyResult] = await Promise.all([
        fetchJson<HealthResponse>(
          `${API_BASE_URL}/continual-learning/health`,
          signal,
        ),
        fetchJson<HistoryResponse>(
          `${API_BASE_URL}/continual-learning/metrics/history`,
          signal,
        ),
      ]);
      setHealth(healthResult);
      setHistory(historyResult.updates);
    } catch (caught) {
      if (signal?.aborted) return;
      setError(
        caught instanceof Error ? caught.message : "Unable to load dashboard",
      );
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadDashboard(controller.signal);
    return () => controller.abort();
  }, [loadDashboard]);

  useEffect(() => {
    if (!evaluation || !["QUEUED", "RUNNING"].includes(evaluation.status)) {
      return;
    }

    const timer = window.setTimeout(async () => {
      try {
        const updated = await fetchJson<EvaluationRun>(
          `${API_BASE_URL}/evaluation/${encodeURIComponent(evaluation.run_id)}`,
        );
        setEvaluation(updated);
      } catch (caught) {
        setEvaluationError(
          caught instanceof Error ? caught.message : "Could not refresh run",
        );
      }
    }, 2000);

    return () => window.clearTimeout(timer);
  }, [evaluation]);

  const startEvaluation = async () => {
    if (!health?.checkpoint?.version || !candidateUri.trim()) return;

    setEvaluationError(null);
    setEvaluation(null);
    try {
      const response = await fetch(`${API_BASE_URL}/evaluation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          active_version: health.checkpoint.version,
          candidate_checkpoint_uri: candidateUri.trim(),
          evaluation_dataset_version: datasetVersion.trim() || "latest",
        }),
      });

      if (!response.ok) throw new Error(await responseError(response));
      setEvaluation((await response.json()) as EvaluationRun);
    } catch (caught) {
      setEvaluationError(
        caught instanceof Error ? caught.message : "Could not start evaluation",
      );
    }
  };

  if (loading) {
    return (
      <main className="min-h-screen bg-slate-950 p-4 text-slate-100 sm:p-6 lg:p-8">
        <div className="mx-auto max-w-7xl space-y-6">
          <div>
            <Skeleton className="h-8 w-72 bg-slate-800" />
            <Skeleton className="mt-2 h-4 w-96 max-w-full bg-slate-800" />
          </div>
          <LoadingDashboard />
        </div>
      </main>
    );
  }

  const accuracy = history.at(-1)?.accuracy;
  const oldVersion = health?.checkpoint?.parent_version;
  const oldRetention = health?.metrics.stability;
  const generalization = health?.metrics.generalization;
  const resistanceEntries = Object.entries(
    evaluation?.breakdown?.attacker_resistance_scores ?? {},
  );

  const chartData = history.map((item) => ({
    version: item.version,
    accuracy: item.accuracy * 100,
    stability: item.stability * 100,
    generalization:
      typeof item.generalization === "number"
        ? item.generalization * 100
        : null,
  }));

  const generatorResults = evaluation?.breakdown?.generators ?? {};
  const forgettingStatus = health?.catastrophic_forgetting_status ?? "LOW";
  const forgettingStyles = {
    LOW: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    MEDIUM: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    HIGH: "border-red-500/30 bg-red-500/10 text-red-300",
  } as const;

  return (
    <main className="min-h-screen bg-slate-950 p-4 text-slate-100 sm:p-6 lg:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm text-cyan-300">
              <Activity className="size-4" />
              Model operations
            </div>
            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
              Continual learning health
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Track retention, generalization, and robustness across model
              updates.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {health?.checkpoint && (
              <Badge
                variant="outline"
                className="border-slate-700 bg-slate-900 text-slate-300"
              >
                Active {health.checkpoint.version}
              </Badge>
            )}
            <Badge
              variant="outline"
              className={forgettingStyles[forgettingStatus]}
            >
              Forgetting: {forgettingStatus}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              disabled={refreshing}
              onClick={() => void loadDashboard()}
              className="border-slate-700 bg-slate-900 text-slate-200 hover:bg-slate-800"
            >
              <RefreshCw
                className={`mr-2 size-4 ${refreshing ? "animate-spin" : ""}`}
              />
              Refresh
            </Button>
          </div>
        </header>

        {error && (
          <div className="flex items-center gap-2 rounded-lg border border-red-900/70 bg-red-950/40 p-4 text-sm text-red-200">
            <AlertCircle className="size-4 shrink-0" />
            {error}
          </div>
        )}

        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            title="Detection accuracy"
            value={percent(accuracy)}
            subtitle={
              health?.checkpoint
                ? `Latest update · ${health.checkpoint.version}`
                : "No model update data"
            }
            icon={Gauge}
            accent="text-cyan-300"
          />
          <MetricCard
            title="Old-generator retention"
            value={percent(oldRetention)}
            subtitle={
              oldVersion ? `Stability · parent ${oldVersion}` : "Stability metric"
            }
            icon={TrendingUp}
            accent="text-emerald-300"
          />
          <MetricCard
            title="Unseen-generator generalization"
            value={percent(generalization)}
            subtitle="Held-out generator performance"
            icon={Sparkles}
            accent="text-violet-300"
          />
          <MetricCard
            title="Scraper resistance efficacy"
            value={
              resistanceEntries.length
                ? scorePercent(
                    resistanceEntries.reduce((sum, [, score]) => sum + score, 0) /
                      resistanceEntries.length,
                  )
                : "—"
            }
            subtitle={
              resistanceEntries.length
                ? `${resistanceEntries.length} attacker tiers · latest evaluation`
                : "Run an evaluation to populate"
            }
            icon={Shield}
            accent="text-amber-300"
          />
        </section>

        <section className="grid gap-6 xl:grid-cols-2">
          <Card className="border-slate-800 bg-slate-900/70">
            <CardHeader>
              <CardTitle className="text-base text-slate-100">
                Historical performance
              </CardTitle>
              <CardDescription className="text-slate-400">
                Accuracy, stability, and generalization by model version.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {chartData.length ? (
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 8, right: 12, left: -16, bottom: 4 }}
                    >
                      <CartesianGrid
                        stroke="#1e293b"
                        strokeDasharray="4 4"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="version"
                        tick={{ fill: "#94a3b8", fontSize: 12 }}
                        axisLine={{ stroke: "#334155" }}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 100]}
                        tickFormatter={(value: number) => `${value}%`}
                        tick={{ fill: "#94a3b8", fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#0f172a",
                          border: "1px solid #334155",
                          borderRadius: 8,
                          color: "#e2e8f0",
                        }}
                        formatter={(value) =>
                          typeof value === "number"
                            ? [`${value.toFixed(1)}%`]
                            : ["—"]
                        }
                      />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        name="Accuracy"
                        stroke="#22d3ee"
                        strokeWidth={2.5}
                        dot={{ r: 3, fill: "#22d3ee" }}
                        connectNulls
                      />
                      <Line
                        type="monotone"
                        dataKey="stability"
                        name="Stability"
                        stroke="#34d399"
                        strokeWidth={2}
                        dot={{ r: 3, fill: "#34d399" }}
                        connectNulls
                      />
                      <Line
                        type="monotone"
                        dataKey="generalization"
                        name="Generalization"
                        stroke="#a78bfa"
                        strokeWidth={2}
                        dot={{ r: 3, fill: "#a78bfa" }}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState message="No historical metrics are available yet." />
              )}
              {chartData.length > 0 &&
                chartData.every((item) => item.generalization === null) && (
                  <p className="mt-2 text-xs text-slate-500">
                    Generalization history is not included by the current API.
                  </p>
                )}
            </CardContent>
          </Card>

          <Card className="border-slate-800 bg-slate-900/70">
            <CardHeader>
              <CardTitle className="text-base text-slate-100">
                Generator performance
              </CardTitle>
              <CardDescription className="text-slate-400">
                Per-generator accuracy from the latest completed evaluation.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              {evaluation?.status === "COMPLETED" ? (
                GENERATORS.map((generator) => {
                  const result = generatorResults[generator];
                  const value = result?.accuracy ?? null;
                  return (
                    <div key={generator} className="space-y-2">
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <span className="text-slate-300">{generator}</span>
                        <span className="font-medium tabular-nums text-slate-100">
                          {percent(value)}
                        </span>
                      </div>
                      <Progress
                        value={value === null ? 0 : value * 100}
                        className="h-2 bg-slate-800 [&>div]:bg-cyan-400"
                      />
                      {result && (
                        <p className="text-xs text-slate-500">
                          {result.sample_count} evaluation samples
                        </p>
                      )}
                    </div>
                  );
                })
              ) : (
                <EmptyState
                  message={
                    evaluation?.status === "FAILED"
                      ? evaluation.error ?? "Evaluation failed."
                      : "Run an evaluation to see generator-level accuracy."
                  }
                />
              )}
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
          <Card className="border-slate-800 bg-slate-900/70">
            <CardHeader>
              <CardTitle className="text-base text-slate-100">
                Scraper resistance tiers
              </CardTitle>
              <CardDescription className="text-slate-400">
                Higher resistance means the recovered image is less similar to
                the original.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              {resistanceEntries.length ? (
                resistanceEntries.map(([attacker, score]) => {
                  const normalized = Math.max(0, Math.min(100, score));
                  return (
                    <div key={attacker} className="space-y-2">
                      <div className="flex items-center justify-between gap-4 text-sm">
                        <span className="text-slate-300">
                          {ATTACKER_LABELS[attacker] ?? attacker}
                        </span>
                        <span className="font-semibold tabular-nums text-slate-100">
                          {scorePercent(score)}
                        </span>
                      </div>
                      <Progress
                        value={normalized}
                        className="h-2 bg-slate-800 [&>div]:bg-amber-400"
                      />
                    </div>
                  );
                })
              ) : (
                <EmptyState message="No attacker resistance results are available." />
              )}
            </CardContent>
          </Card>

          <Card className="border-slate-800 bg-slate-900/70">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base text-slate-100">
                <FlaskConical className="size-4 text-violet-300" />
                Evaluate candidate checkpoint
              </CardTitle>
              <CardDescription className="text-slate-400">
                Compare a candidate against the active model and evaluation
                dataset.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="candidate-uri" className="text-slate-300">
                  Candidate checkpoint URI
                </Label>
                <Input
                  id="candidate-uri"
                  value={candidateUri}
                  onChange={(event) => setCandidateUri(event.target.value)}
                  placeholder="s3://checkpoints/model-v4.3.pt"
                  className="border-slate-700 bg-slate-950 text-slate-100 placeholder:text-slate-600"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dataset-version" className="text-slate-300">
                  Evaluation dataset version
                </Label>
                <Input
                  id="dataset-version"
                  value={datasetVersion}
                  onChange={(event) => setDatasetVersion(event.target.value)}
                  placeholder="latest"
                  className="border-slate-700 bg-slate-950 text-slate-100"
                />
              </div>

              <Button
                className="w-full"
                disabled={
                  !health?.checkpoint?.version ||
                  !candidateUri.trim() ||
                  evaluation?.status === "QUEUED" ||
                  evaluation?.status === "RUNNING"
                }
                onClick={() => void startEvaluation()}
              >
                {evaluation?.status === "QUEUED" ||
                evaluation?.status === "RUNNING" ? (
                  <Clock3 className="mr-2 size-4 animate-pulse" />
                ) : (
                  <FlaskConical className="mr-2 size-4" />
                )}
                {evaluation?.status === "QUEUED" ||
                evaluation?.status === "RUNNING"
                  ? `Evaluation ${evaluation.status.toLowerCase()}`
                  : "Run evaluation"}
              </Button>

              {evaluation?.status === "COMPLETED" && (
                <div className="flex items-center gap-2 rounded-md border border-emerald-800/70 bg-emerald-950/30 p-3 text-sm text-emerald-300">
                  <CheckCircle2 className="size-4 shrink-0" />
                  Evaluation completed
                  {evaluation.completed_at
                    ? ` · ${new Date(evaluation.completed_at).toLocaleTimeString()}`
                    : ""}
                </div>
              )}
              {(evaluationError || evaluation?.error) && (
                <div className="flex items-start gap-2 rounded-md border border-red-900/70 bg-red-950/30 p-3 text-sm text-red-300">
                  <AlertCircle className="mt-0.5 size-4 shrink-0" />
                  {evaluationError ?? evaluation?.error}
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      </div>
    </main>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex min-h-36 items-center justify-center rounded-lg border border-dashed border-slate-800 px-5 text-center text-sm text-slate-500">
      {message}
    </div>
  );
}
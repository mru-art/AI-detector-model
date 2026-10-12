import { USE_MOCK } from "@/lib/analysis";

// Our own shape for the dashboard. The backend contract isn't agreed yet,
// so this is where an adapter goes once it is.

export type Column = {
  id: string;
  label: string;
  /** real = real images, generator = trained generator, heldout = never trained on */
  kind: "real" | "generator" | "heldout";
};

export type Experience = {
  id: string;
  generator: string;
  /** which column this experience trains */
  columnId: string;
};

export type DashboardData = {
  columns: Column[];
  experiences: Experience[];
  /** matrix[row][col]: accuracy (0–100) after experience `row`, on column `col` */
  matrix: number[][];
  /** True while the numbers are made up locally, not from the model. */
  isMock: boolean;
};

export type ExperienceSummary = {
  id: string;
  generator: string;
  learned: number;
  now: number;
  change: number;
};

export type Summary = {
  averageAccuracy: number;
  heldOut: { label: string; accuracy: number } | null;
  /** Backward transfer: average change on earlier generators. Negative = forgetting. */
  bwt: number | null;
  perExperience: ExperienceSummary[];
};

const mean = (values: number[]): number =>
  values.reduce((sum, v) => sum + v, 0) / values.length;

/** Has the model been trained on column `col` by the time of row `row`? */
export function isTrained(data: DashboardData, row: number, col: number): boolean {
  const column = data.columns[col];
  if (column.kind === "real") return true;
  if (column.kind === "heldout") return false;
  const index = data.experiences.findIndex((e) => e.columnId === column.id);
  return index !== -1 && index <= row;
}

export function summarize(data: DashboardData): Summary {
  const last = data.matrix.length - 1;

  const trained = data.columns
    .map((_, col) => col)
    .filter((col) => isTrained(data, last, col));
  const averageAccuracy = mean(trained.map((col) => data.matrix[last][col]));

  const heldCol = data.columns.findIndex((c) => c.kind === "heldout");
  const heldOut =
    heldCol === -1
      ? null
      : { label: data.columns[heldCol].label, accuracy: data.matrix[last][heldCol] };

  const perExperience = data.experiences.map((experience, row) => {
    const col = data.columns.findIndex((c) => c.id === experience.columnId);
    const learned = data.matrix[row][col];
    const now = data.matrix[last][col];
    return {
      id: experience.id,
      generator: experience.generator,
      learned,
      now,
      change: now - learned,
    };
  });

  const earlier = perExperience.slice(0, -1);
  const bwt = earlier.length ? mean(earlier.map((p) => p.change)) : null;

  return { averageAccuracy, heldOut, bwt, perExperience };
}

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/** MOCK: made-up numbers in the shape of the project's accuracy matrix. */
async function mockDashboard(signal?: AbortSignal): Promise<DashboardData> {
  await wait(500, signal);
  return {
    columns: [
      { id: "real", label: "Real", kind: "real" },
      { id: "sd", label: "Stable Diffusion", kind: "generator" },
      { id: "mj", label: "Midjourney", kind: "generator" },
      { id: "dalle", label: "DALL-E", kind: "generator" },
      { id: "flux", label: "FLUX", kind: "heldout" },
    ],
    experiences: [
      { id: "E1", generator: "Stable Diffusion", columnId: "sd" },
      { id: "E2", generator: "Midjourney", columnId: "mj" },
      { id: "E3", generator: "DALL-E", columnId: "dalle" },
    ],
    matrix: [
      [97, 96, 71, 64, 58],
      [98, 95, 96, 74, 66],
      [98, 93, 95, 94, 71],
    ],
    isMock: true,
  };
}

export async function fetchDashboard(signal?: AbortSignal): Promise<DashboardData> {
  if (USE_MOCK) return mockDashboard(signal);
  throw new Error("The dashboard API isn’t connected yet.");
}

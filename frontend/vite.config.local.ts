// Local-only Vite config (excluded from git via .git/info/exclude).
// Run:  npx vite --config vite.config.local.ts
//
// * Proxies /api/* to the FastAPI backend, so the browser never hits CORS.
// * Swaps "@/lib/analysis" for local_dev/frontend/analysis.ts, which calls
//   the real /detect endpoint, without touching the repo's source.
import { fileURLToPath, URL } from "node:url";
import { defineConfig, mergeConfig } from "vite";
import base from "./vite.config";

const backend = process.env.BACKEND_URL ?? "http://localhost:8000";

export default mergeConfig(
  base,
  defineConfig({
    resolve: {
      alias: [
        {
          find: /^@\/lib\/analysis$/,
          replacement: fileURLToPath(new URL("../local_dev/frontend/analysis.ts", import.meta.url)),
        },
      ],
    },
    server: {
      fs: { allow: [".."] },
      proxy: {
        "/api": {
          target: backend,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
  }),
);

// Local dev config: run the frontend against a backend on this PC.
// Run:  npx vite --config vite.config.local.ts
//
// * Turns off mock mode and sends API calls to /api.
// * Proxies /api/* to the FastAPI backend, so the browser never hits CORS.
import { defineConfig, mergeConfig } from "vite";
import base from "./vite.config";

const backend = process.env.BACKEND_URL ?? "http://localhost:8000";

// Vite reads VITE_* variables from process.env after loading this file.
process.env.VITE_USE_MOCK ??= "false";
process.env.VITE_API_BASE_URL ??= "/api";

export default mergeConfig(
  base,
  defineConfig({
    server: {
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

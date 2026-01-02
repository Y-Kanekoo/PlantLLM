import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/diagnose": "http://localhost:8000",
      "/history": "http://localhost:8000",
      "/chat": "http://localhost:8000",
      "/models": "http://localhost:8000",
      "/model-status": "http://localhost:8000",
      "/clear-history": "http://localhost:8000",
      "/uploads": "http://localhost:8000"
    }
  }
});

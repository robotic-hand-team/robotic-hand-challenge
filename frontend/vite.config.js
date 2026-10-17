import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // 127.0.0.1, not localhost: Node resolves localhost to ::1 first, uvicorn listens on IPv4 only.
    proxy: { "/api": process.env.VITE_API_URL || "http://127.0.0.1:8000" },
    // Windows: "c:\" vs "C:\" drive casing makes Vite reject its own dep cache with 403.
    fs: { strict: false },
  },
  optimizeDeps: { include: ["three", "@react-three/fiber", "@react-three/drei"] },
});

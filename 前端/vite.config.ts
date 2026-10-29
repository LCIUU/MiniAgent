import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 开发环境代理到 FastAPI，避免 CORS
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true, // WebSocket 代理（控制通道）
      },
    },
  },
});

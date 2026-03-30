import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";
import fs from "fs";
import { randomUUID } from "crypto";

const UPLOADS_DIR = path.resolve(__dirname, "dev/uploads");

function uploadPlugin(): Plugin {
  return {
    name: "upload-middleware",
    configureServer(server) {
      server.middlewares.use("/upload", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("Method not allowed");
          return;
        }

        const contentType = req.headers["content-type"] ?? "";
        const boundary = contentType.split("boundary=")[1];
        if (!boundary) {
          res.statusCode = 400;
          res.end("Missing multipart boundary");
          return;
        }

        const chunks: Buffer[] = [];
        req.on("data", (chunk: Buffer) => chunks.push(chunk));
        req.on("end", () => {
          const body = Buffer.concat(chunks);
          const parts = body.toString("binary").split(`--${boundary}`);

          for (const part of parts) {
            const headerEnd = part.indexOf("\r\n\r\n");
            if (headerEnd === -1) continue;
            const headers = part.slice(0, headerEnd);
            const filenameMatch = headers.match(/filename="(.+?)"/);
            if (!filenameMatch) continue;

            const originalName = filenameMatch[1];
            const ext = path.extname(originalName);
            const savedName = `${randomUUID()}${ext}`;
            const content = part.slice(headerEnd + 4, part.lastIndexOf("\r\n"));

            fs.mkdirSync(UPLOADS_DIR, { recursive: true });
            const filePath = path.join(UPLOADS_DIR, savedName);
            fs.writeFileSync(filePath, content, "binary");

            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ path: filePath }));
            return;
          }

          res.statusCode = 400;
          res.end("No file found in upload");
        });
      });
    },
  };
}

const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
const backendWs = backendUrl.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react(), tailwindcss(), uploadPlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      external: ["@tauri-apps/plugin-dialog"],
    },
  },
  optimizeDeps: {
    exclude: ["@tauri-apps/plugin-dialog"],
  },
  server: {
    host: true,
    proxy: {
      "/graph": {
        target: backendUrl,
        changeOrigin: true,
      },
      "/metrics": {
        target: backendUrl,
        changeOrigin: true,
      },
      "/component": {
        target: backendUrl,
        changeOrigin: true,
      },
      "/projects": {
        target: backendUrl,
        changeOrigin: true,
      },
      "/project": {
        target: backendUrl,
        changeOrigin: true,
      },
      "/env": {
        target: backendUrl,
        changeOrigin: true,
      },
      "/logs": {
        target: backendUrl,
        changeOrigin: true,
      },
      "/ui/ws": {
        target: backendWs,
        ws: true,
      },
    },
  },
});

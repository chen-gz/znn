import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, type Plugin } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

/**
 * Serves and bundles znn's canonical model graph schema and exported JSON graphs
 * under `/tools/visualizer/*` directly from `src/nn/` and `examples/` without
 * duplicating files inside `web/`.
 */
function znnGraphAssetsPlugin(): Plugin {
  const resolveAssetPath = (urlPath: string): string | null => {
    const clean = urlPath.split("?")[0];
    if (clean === "/tools/visualizer/model_graph.schema.json") {
      return path.join(repoRoot, "src/nn/model_graph.schema.json");
    }
    if (clean === "/tools/visualizer/sample_model_graph.json") {
      return path.join(repoRoot, "examples/sample_model_graph.json");
    }
    if (clean === "/tools/visualizer/minimal_model_graph.json") {
      return path.join(repoRoot, "examples/minimal_model_graph.json");
    }
    const prefix = "/tools/visualizer/models/";
    if (clean.startsWith(prefix) && clean.endsWith(".json")) {
      const base = path.basename(clean);
      return path.join(repoRoot, "examples/models", base);
    }
    return null;
  };

  return {
    name: "znn-graph-assets",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const filePath = req.url ? resolveAssetPath(req.url) : null;
        if (filePath && fs.existsSync(filePath)) {
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(fs.readFileSync(filePath));
          return;
        }
        next();
      });
    },
    generateBundle() {
      const emit = (fileName: string, absPath: string) => {
        if (fs.existsSync(absPath)) {
          this.emitFile({
            type: "asset",
            fileName,
            source: fs.readFileSync(absPath, "utf-8"),
          });
        }
      };
      emit("tools/visualizer/model_graph.schema.json", path.join(repoRoot, "src/nn/model_graph.schema.json"));
      emit("tools/visualizer/sample_model_graph.json", path.join(repoRoot, "examples/sample_model_graph.json"));
      emit("tools/visualizer/minimal_model_graph.json", path.join(repoRoot, "examples/minimal_model_graph.json"));
      const modelsDir = path.join(repoRoot, "examples/models");
      if (fs.existsSync(modelsDir)) {
        for (const entry of fs.readdirSync(modelsDir)) {
          if (entry.endsWith(".json")) {
            emit(`tools/visualizer/models/${entry}`, path.join(modelsDir, entry));
          }
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [tailwindcss(), znnGraphAssetsPlugin()],
  server: {
    fs: {
      allow: [repoRoot],
    },
  },
});

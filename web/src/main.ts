import "./style.css";
import defaultModelGraph from "../../examples/sample_model_graph.json";
import { renderSchemaGuide } from "./schema-guide";
import { initVisualizer, type ModelHierarchyGraph } from "./scripts/visualizer";

function setupThemeToggle(): void {
  const saved = localStorage.getItem("znn-vis-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const isDark = saved ? saved === "dark" : prefersDark;
  document.documentElement.classList.toggle("dark", isDark);

  const toggleBtn = document.getElementById("theme-toggle-btn");
  toggleBtn?.addEventListener("click", () => {
    const nextDark = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", nextDark);
    localStorage.setItem("znn-vis-theme", nextDark ? "dark" : "light");
  });
}

function bootstrap(): void {
  setupThemeToggle();
  renderSchemaGuide(document.getElementById("schema-guide-root"));
  initVisualizer(defaultModelGraph as unknown as ModelHierarchyGraph);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}

import schemaJson from "../../src/nn/model_graph.schema.json";
import minimalGraph from "../../examples/minimal_model_graph.json";
import { schemaSections, type JsonSchema } from "./utils/json-schema";
import DOMPurify from "dompurify";

const SCHEMA_URL = "/tools/visualizer/model_graph.schema.json";
const MINIMAL_URL = "/tools/visualizer/minimal_model_graph.json";
const SAMPLE_URL = "/tools/visualizer/sample_model_graph.json";

function esc(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function renderSchemaGuide(container: HTMLElement | null): void {
  if (!container) return;
  const schema = schemaJson as JsonSchema;
  const sections = schemaSections(schema);
  const fieldCount = sections.reduce((n, s) => n + s.fields.length, 0);
  const schemaText = JSON.stringify(schema, null, 2);
  const templateText = JSON.stringify(minimalGraph, null, 2);
  const version = schema.properties?.version?.const ?? "";
  const anchor = (name: string) => `schema-def-${name}`;

  const linkBtn =
    "px-2.5 py-1 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded text-[11px] font-semibold hover:bg-[var(--bg-card-hover)] no-underline text-[var(--color-text-primary)]";
  const card = "bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-4.5";
  const pre =
    "bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg p-3 font-mono text-xs text-[var(--color-text-primary)] overflow-x-auto max-h-80";

  const navLinks = sections
    .map(
      (s) =>
        `<a href="#${anchor(s.name)}" class="px-2 py-0.5 rounded-full border border-[var(--border-color)] font-mono text-[var(--color-accent)] no-underline hover:bg-[var(--bg-card-hover)]">${esc(s.name)}</a>`,
    )
    .join("");

  const sectionCards = sections
    .map((s) => {
      const rows = s.fields
        .map((f) => {
          const typeCell = f.ref
            ? `<a href="#${anchor(f.ref)}" class="text-amber-500 underline decoration-dotted">${esc(f.type)}</a>`
            : esc(f.type);
          const reqCell = f.required ? "yes" : `<span class="text-[var(--theme-prose-lead)]">optional</span>`;
          return `<tr class="border-b border-[var(--border-color)] hover:bg-black/5 dark:hover:bg-white/5 align-top">
            <td class="p-2 font-mono text-[var(--color-accent)] font-semibold whitespace-nowrap">${esc(f.name)}</td>
            <td class="p-2 font-mono text-amber-500">${typeCell}</td>
            <td class="p-2 whitespace-nowrap">${reqCell}</td>
            <td class="p-2">${esc(f.description)}</td>
          </tr>`;
        })
        .join("");

      return `<div id="${anchor(s.name)}" class="${card} scroll-mt-20">
        <div class="text-sm font-bold text-[var(--color-accent)] mb-1 font-mono">${esc(s.name)}</div>
        <p class="text-xs text-[var(--theme-prose-lead)] mb-3">${esc(s.description)}</p>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left border-collapse">
            <thead>
              <tr class="border-b border-[var(--border-color)] text-[11px] uppercase text-[var(--theme-prose-lead)] bg-[var(--bg-secondary)]">
                <th class="p-2">Field</th>
                <th class="p-2">Type</th>
                <th class="p-2">Required</th>
                <th class="p-2">Description</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>`;
    })
    .join("");

  container.innerHTML = DOMPurify.sanitize(`<details id="schema-guide" class="mb-6 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl overflow-hidden group">
    <summary class="bg-[var(--bg-card)] px-4 py-3 cursor-pointer flex items-center gap-3 list-none select-none hover:bg-[var(--bg-card-hover)] transition-colors">
      <span class="transition-transform group-open:rotate-90 inline-block text-[10px] text-[var(--color-accent)]">▶</span>
      <span class="text-base">📖</span>
      <div>
        <div class="text-sm font-bold text-[var(--color-accent)]">JSON Format Guide</div>
        <div class="text-xs text-[var(--theme-prose-lead)]">
          ${esc(schema.title)} schema ${esc(version)}: ${sections.length} object types, ${fieldCount} fields, JSON Schema and a minimal template
        </div>
      </div>
    </summary>
    <div class="p-5 flex flex-col gap-5 text-[13px] leading-relaxed text-[var(--color-text-primary)]">
      <div class="${card}">
        <div class="text-sm font-bold text-[var(--color-accent)] mb-2">📌 Overview</div>
        <p class="text-xs text-[var(--theme-prose-lead)] mb-3">${esc(schema.description)}</p>
        <p class="text-xs text-[var(--theme-prose-lead)] mb-3">
          The format is defined by a JSON Schema (draft 2020-12). Every object is closed (<code>additionalProperties: false</code>), and
          uploaded files are validated against it before rendering. The tables below are generated from the schema.
        </p>
        <div class="flex flex-wrap items-center gap-2">
          <a class="${linkBtn}" href="${SCHEMA_URL}" download>⬇ model_graph.schema.json</a>
          <a class="${linkBtn}" href="${MINIMAL_URL}" download>⬇ minimal_model_graph.json</a>
          <a class="${linkBtn}" href="${SAMPLE_URL}" download>⬇ sample_model_graph.json</a>
        </div>
        <nav class="mt-3 flex flex-wrap gap-1.5 text-[11px]" aria-label="Schema object types">${navLinks}</nav>
      </div>
      ${sectionCards}
      <div class="${card}">
        <div class="flex items-center justify-between mb-2 gap-2">
          <span class="text-sm font-bold text-[var(--color-accent)]">📄 Minimal JSON Template</span>
          <button
            class="px-2.5 py-1 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded text-[11px] font-semibold hover:bg-[var(--bg-card-hover)] cursor-pointer"
            id="btn-copy-template">📋 Copy JSON</button>
        </div>
        <p class="text-xs text-[var(--theme-prose-lead)] mb-2">
          A single <code>Linear</code> layer exported by znn (<code>zig build run-report</code>), so every field is present with real values.
        </p>
        <pre class="${pre}"><code id="schema-template-code">${esc(templateText)}</code></pre>
      </div>
      <details class="${card} group/schema">
        <summary class="flex items-center justify-between gap-2 cursor-pointer list-none select-none">
          <span class="text-sm font-bold text-[var(--color-accent)] flex items-center gap-2">
            <span class="transition-transform group-open/schema:rotate-90 inline-block text-[10px]">▶</span>
            🧾 JSON Schema (model_graph.schema.json)
          </span>
          <button
            class="px-2.5 py-1 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded text-[11px] font-semibold hover:bg-[var(--bg-card-hover)] cursor-pointer"
            id="btn-copy-schema">📋 Copy Schema</button>
        </summary>
        <pre class="${pre} mt-3"><code id="schema-json-code">${esc(schemaText)}</code></pre>
      </details>
    </div>
  </details>`);
}

/**
 * src/scripts/visualizer.ts
 *
 * Client-side controller for the ZNN Neural Network Architecture Explorer.
 * Renders ModelHierarchyGraph schema 2.0: every module carries its own scoped local
 * graph (`ports`, `flow_nodes`, `edges`), which drives the card pipeline, the Mermaid
 * view and the inspector. Graph logic lives in `src/utils/model-graph.ts`.
 * Specification: doc/visualization-model-edge-design.md.
 */

import mermaid from "mermaid";
import katex from "katex";
import DOMPurify from "dompurify";
import { getMermaidConfig } from "../utils/mermaid-theme";
import { formatNumber, formatBytes } from "../utils/format";
import {
  buildLocalGraph,
  buildMermaidLocalGraph,
  collectSubtree,
  edgeShapeLabel,
  findModuleByPath,
  hasLocalGraph,
  isAddNode,
  layoutPipeline,
  listModules,
  pipelineMinWidthRem,
  pipelineUnitMinWidthRem,
  resolveCanonical,
  type EdgeData,
  type FlowNode,
  type LocalGraph,
  type ModelHierarchyGraph,
  type ModuleNode,
  type NodeData,
  type NodeKind,
  type OpData,
  type PipelineRow,
  type PipelineUnit,
} from "../utils/model-graph";
import { routeEdges, type EdgeRouteKind, type Rect } from "../utils/flow-edges";
import { validateJson, type JsonSchema } from "../utils/json-schema";
import modelGraphSchema from "../../../src/nn/model_graph.schema.json";

const MODEL_GRAPH_SCHEMA = modelGraphSchema as JsonSchema;

export type { EdgeData, FlowNode, ModelHierarchyGraph, ModuleNode, NodeData, NodeKind, OpData };

// State
let RAW_GRAPH: ModelHierarchyGraph | null = null;
let NODES_DATA: NodeData[] = [];
let OPS_DATA: OpData[] = [];
let FORMULAS_DATA: Record<string, string> = {};
let DAG_SCOPE = "";
let MERMAID_SCOPE = "";
let currentKindFilter = "all";
let currentQuery = "";
let mermaidRenderSeq = 0;
let mermaidQueue: Promise<void> = Promise.resolve();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function esc(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function rootModule(): ModuleNode | null {
  return RAW_GRAPH ? RAW_GRAPH.root : null;
}

function defaultScope(): string {
  const scope = RAW_GRAPH?.default_scope ?? "";
  const mod = findModuleByPath(rootModule(), scope);
  return mod && hasLocalGraph(mod) ? scope : "";
}

function moduleLabel(mod: ModuleNode): string {
  return mod.path ? mod.path : "root";
}

/** Visual identity of a flow node, derived only from `kind` / `module_type` / `op_type`. */
function flowVisual(n: FlowNode): { icon: string; badge: string; tone: string } {
  switch (n.kind) {
    case "port_in":
      return { icon: "📥", badge: "Input port", tone: "text-emerald-500 bg-emerald-500/10" };
    case "port_out":
      return { icon: "📤", badge: "Output port", tone: "text-sky-500 bg-sky-500/10" };
    case "buffer":
      return { icon: "🧱", badge: "Buffer", tone: "text-slate-500 bg-slate-500/10" };
    case "module":
      return { icon: "🧩", badge: n.module_type || "Module", tone: "text-[var(--color-accent)] bg-[var(--bg-secondary)]" };
    default:
      return isAddNode(n)
        ? { icon: "⊕", badge: "Add", tone: "text-amber-500 bg-amber-500/10" }
        : { icon: "⚡", badge: n.op_type || "Op", tone: "text-purple-400 bg-purple-400/10" };
  }
}

const NODE_KIND_ICON: Record<NodeKind, string> = {
  Param: "⚖️",
  Input: "📥",
  Buffer: "🧱",
  Activation: "📦",
};

function nodeKindIcon(kind: NodeKind): string {
  return NODE_KIND_ICON[kind];
}

// ---------------------------------------------------------------------------
// Core Loader
// ---------------------------------------------------------------------------

/** Loads a graph; `source` names where it came from (bundled sample or an uploaded file name). */
export function loadGraphData(graphObj: ModelHierarchyGraph, source: string = "bundled sample") {
  if (!graphObj || !graphObj.root) {
    throw new Error("Invalid graph data: missing root node");
  }
  RAW_GRAPH = graphObj;
  const all = collectSubtree(graphObj.root);
  NODES_DATA = all.nodes;
  OPS_DATA = all.ops;
  FORMULAS_DATA = {};
  for (const m of listModules(graphObj.root)) {
    if (m.formula) FORMULAS_DATA[m.path || "root"] = m.formula;
  }
  for (const op of OPS_DATA) {
    if (op.formula && op.name) FORMULAS_DATA[op.name] = op.formula;
  }
  DAG_SCOPE = defaultScope();
  MERMAID_SCOPE = DAG_SCOPE;
  CURRENT_DETAILS_KEY = null;
  DETAILS_HISTORY.length = 0;
  setLoadStatus("");

  const infoEl = document.getElementById("loaded-model-info");
  if (infoEl) infoEl.textContent = loadedModelLabel(graphObj, source);

  const selectEl = document.getElementById("details-select-module") as HTMLSelectElement | null;
  if (selectEl) selectEl.innerHTML = "";

  initKpis();
  renderDagView();
  renderTree();
  renderMermaidView();
  renderDedicatedInspectorView();
}

/** e.g. `gpt (GPT) · schema 2.0 · bundled sample` */
function loadedModelLabel(graphObj: ModelHierarchyGraph, source: string): string {
  const main = findModuleByPath(graphObj.root, DAG_SCOPE) || graphObj.root;
  const name = moduleLabel(main);
  const type = main.module_type ? ` (${main.module_type})` : "";
  return `${name}${type} · schema ${graphObj.version} · ${source}`;
}

export function initKpis() {
  const sum = RAW_GRAPH?.summary || {};
  const count = (kind: string) => NODES_DATA.filter((n) => n.kind === kind).length;
  const totalParams = sum.total_params ?? NODES_DATA.filter((n) => n.kind === "Param").reduce((a, n) => a + (n.elements || 0), 0);
  const totalBytes = sum.total_bytes ?? NODES_DATA.reduce((a, n) => a + (n.bytes || 0), 0);
  const paramNodes = sum.param_nodes ?? count("Param");
  const inputNodes = sum.input_nodes ?? count("Input");
  const bufferNodes = sum.buffer_nodes ?? count("Buffer");
  const actNodes = sum.activation_nodes ?? count("Activation");
  const totalNodes = sum.total_nodes ?? NODES_DATA.length;

  const set = (id: string, text: string) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };
  set("kpi-total-params", formatNumber(totalParams));
  set("kpi-memory-mb", `~${(totalBytes / (1024 * 1024)).toFixed(2)} MB Memory Footprint`);
  set("kpi-param-nodes", formatNumber(paramNodes));
  set("kpi-init-stats", `${sum.auto_graph_count || 0} Auto-Graph / ${sum.custom_init_count || 0} Custom`);
  set("kpi-total-nodes", formatNumber(totalNodes));
  set("kpi-node-breakdown", `${inputNodes} Inputs · ${actNodes} Activations`);
  set("kpi-buffers", formatNumber(bufferNodes));
}

// ---------------------------------------------------------------------------
// Scoped card pipeline (DAG view and inspector)
// ---------------------------------------------------------------------------

function renderEdgeLine(mod: ModuleNode, e: EdgeData): string {
  const transforms = (e.transforms || [])
    .map((t) => `<span class="px-1 rounded bg-[var(--bg-secondary)] border border-[var(--border-color)]">${esc(t.split(" ")[0])}</span>`)
    .join(" ");
  const kindTag = e.kind === "buffer" ? `<span class="text-slate-500">buffer</span>` : "";
  return `
    <div class="flex items-center gap-1 flex-wrap text-[9.5px] font-mono text-[var(--theme-prose-lead)]" title="${esc((e.transforms || []).join("\n"))}">
      <span>←</span>
      <span class="font-semibold text-[var(--color-text-primary)] cursor-pointer hover:underline" data-inspect="${esc(resolveCanonical(mod, e.from))}">${esc(e.from)}</span>
      <span>${esc(edgeShapeLabel(e))}</span>
      ${transforms} ${kindTag}
    </div>`;
}

function renderNodeCard(mod: ModuleNode, g: LocalGraph, id: string, compact: boolean): string {
  const n = g.nodes.get(id)!;
  const vis = flowVisual(n);
  const canonical = resolveCanonical(mod, id);
  const inEdges = (g.incoming.get(id) || []).filter((e) => !e.is_skip);
  const fanIn = inEdges.length > 1 && !isAddNode(n) ? `<span class="text-[9px] font-mono text-[var(--theme-prose-lead)]">${inEdges.length} inputs</span>` : "";
  const child = n.kind === "module" ? findModuleByPath(rootModule(), n.ref) : null;
  const enter =
    child && hasLocalGraph(child)
      ? `<span class="px-1.5 py-0.5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded text-[9.5px] font-medium" title="${esc(child.path)} has an internal graph">⤵ Graph</span>`
      : "";
  const shape = n.kind === "port_in" || n.kind === "port_out" || n.kind === "buffer" ? n.shape || "" : "";

  return `
    <div class="w-full bg-[var(--bg-card)] border ${isAddNode(n) ? "border-amber-500/50" : "border-[var(--border-color)]"} hover:border-[var(--color-accent)] rounded-xl ${compact ? "p-2" : "p-3"} cursor-pointer transition-all shadow-xs" data-inspect="${esc(canonical)}" data-flow-id="${esc(id)}">
      <div class="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
        <div class="flex items-center gap-1.5 min-w-0">
          <span class="text-xs">${vis.icon}</span>
          <span class="font-mono text-xs font-bold text-[var(--color-accent)] truncate">${esc(id)}</span>
          ${fanIn}
        </div>
        <div class="flex flex-wrap items-center gap-1.5 min-w-0 max-w-full">
          ${shape ? `<span class="text-[9.5px] font-mono text-[var(--theme-prose-lead)]">${esc(shape)}</span>` : ""}
          <span class="text-[9.5px] font-semibold px-1.5 py-0.5 rounded ${vis.tone}">${esc(vis.badge)}</span>
          ${enter}
        </div>
      </div>
      <div class="text-[9.5px] font-mono text-[var(--theme-prose-lead)] truncate mt-0.5" title="${esc(canonical)}">${esc(canonical)}</div>
      ${inEdges.length > 0 ? `<div class="mt-1.5 pt-1 border-t border-[var(--border-color)]/50 flex flex-col gap-0.5">${inEdges.map((e) => renderEdgeLine(mod, e)).join("")}</div>` : ""}
    </div>`;
}

function renderResidualUnit(mod: ModuleNode, g: LocalGraph, unit: Extract<PipelineUnit, { type: "residual" }>, compact: boolean): string {
  const skip = unit.skip;
  // The frame holds F(x); its left padding is the skip lane (see routeEdges). The merge node
  // sits below the frame with the same left edge as the skip source, so the skip edge is a
  // straight vertical line down the lane from the source into the merge node.
  return `
    <div class="flex flex-col items-stretch gap-7 w-full">
      <div class="relative bg-amber-500/5 border border-dashed border-amber-500/40 rounded-xl py-3 pr-3 pl-7 w-full" data-flow-block="${esc(unit.target)}">
        <div class="flex items-center justify-between gap-2 mb-2 text-[9px] font-mono">
          <span class="font-bold uppercase text-amber-500">⚡ Residual · skip from ${esc(skip.from)}</span>
          <span class="text-[var(--theme-prose-lead)]">${esc(edgeShapeLabel(skip))} · ${unit.branchIds.length} nodes in F(x)</span>
        </div>
        <div class="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg p-2.5">
          <div class="text-[9px] font-bold uppercase text-purple-400 mb-2">⚙️ Transform F(x)</div>
          <div class="flex flex-col items-stretch gap-7">${renderRows(mod, g, unit.branch, true)}</div>
        </div>
      </div>
      ${renderNodeCard(mod, g, unit.target, compact)}
    </div>`;
}

function renderRows(mod: ModuleNode, g: LocalGraph, rows: PipelineRow[], compact: boolean): string {
  return rows
    .map((row) => {
      const cells = row.units.map((u) => (u.type === "node" ? renderNodeCard(mod, g, u.id, compact) : renderResidualUnit(mod, g, u, compact)));
      const columns = row.units.map((u) => `minmax(${pipelineUnitMinWidthRem(u)}rem, 1fr)`).join(" ");
      const body =
        cells.length === 1
          ? cells[0]
          : `<div class="grid gap-x-2 w-full items-start" style="grid-template-columns: ${columns};">${cells.join("")}</div>`;
      return `<div class="w-full">${body}</div>`;
    })
    .join("");
}

function renderLocalPipelineHtml(mod: ModuleNode): string {
  if (!hasLocalGraph(mod)) return "";
  const g = buildLocalGraph(mod);
  const rows = layoutPipeline(g);
  const skipCount = g.edges.filter((e) => e.is_skip).length;
  const minWidth = pipelineMinWidthRem(rows);
  const maxWidth = Math.max(36, Math.round(minWidth * 1.3 * 10) / 10);
  return `
    <div class="mb-5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 shadow-xs w-full">
      <div class="flex items-center justify-between pb-3 mb-3 border-b border-[var(--border-color)]">
        <div class="text-xs font-bold uppercase tracking-wider text-[var(--color-accent)] flex items-center gap-2">
          <span>🔀</span>
          <span>${esc(moduleLabel(mod))} · ${esc(mod.module_type || "Module")} · Local Data Flow</span>
        </div>
        <span class="text-[11px] font-mono text-[var(--theme-prose-lead)]">${g.edges.length} edges${skipCount ? ` · ${skipCount} residual` : ""}</span>
      </div>
      <div class="overflow-x-auto">
        <div class="mx-auto pl-3 pr-16 py-1" style="min-width: ${minWidth + 4.75}rem; max-width: ${maxWidth + 4.75}rem;">
          <div class="relative w-full" data-flow-scope="${esc(mod.path)}">
            <div class="flex flex-col items-stretch gap-7">${renderRows(mod, g, rows, false)}</div>
            <svg class="flow-edges absolute inset-0 w-full h-full pointer-events-none overflow-visible" aria-hidden="true"></svg>
          </div>
        </div>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Edge drawing: one SVG path per real edge of the local graph
// ---------------------------------------------------------------------------

const EDGE_STYLE: Record<EdgeRouteKind, { color: string; dash: string; width: number }> = {
  data: { color: "var(--theme-prose-lead)", dash: "", width: 1.4 },
  skip: { color: "#f59e0b", dash: "5 4", width: 1.6 },
  buffer: { color: "#64748b", dash: "2 3", width: 1.4 },
};

let flowSvgSeq = 0;

const flowResizeObserver =
  typeof ResizeObserver === "undefined"
    ? null
    : new ResizeObserver((entries) => {
        for (const entry of entries) {
          const el = entry.target as HTMLElement;
          if (!el.isConnected) {
            flowResizeObserver?.unobserve(el);
            continue;
          }
          drawFlowEdges(el);
        }
      });

/** Draw the edges of every pipeline under `root` and keep them in sync with layout changes. */
function mountFlowEdges(root: HTMLElement | null) {
  root?.querySelectorAll<HTMLElement>("[data-flow-scope]").forEach((el) => {
    drawFlowEdges(el);
    flowResizeObserver?.observe(el);
  });
}

function drawFlowEdges(scopeEl: HTMLElement) {
  const svg = scopeEl.querySelector<SVGSVGElement>(":scope > svg.flow-edges");
  const mod = findModuleByPath(rootModule(), scopeEl.dataset.flowScope || "");
  const origin = scopeEl.getBoundingClientRect();
  if (!svg || !mod || origin.width === 0) return;

  const rel = (el: Element): Rect => {
    const r = el.getBoundingClientRect();
    return { x: r.left - origin.left, y: r.top - origin.top, w: r.width, h: r.height };
  };
  const cards = new Map<string, Rect>();
  scopeEl.querySelectorAll<HTMLElement>("[data-flow-id]").forEach((el) => cards.set(el.dataset.flowId || "", rel(el)));
  const blocks = new Map<string, Rect>();
  scopeEl.querySelectorAll<HTMLElement>("[data-flow-block]").forEach((el) => blocks.set(el.dataset.flowBlock || "", rel(el)));

  const routes = routeEdges(buildLocalGraph(mod).edges, { cards, blocks, bounds: { x: 0, y: 0, w: origin.width, h: origin.height } });
  const seq = ++flowSvgSeq;
  const markerId = (kind: EdgeRouteKind) => `flow-arrow-${seq}-${kind}`;
  const defs = (Object.keys(EDGE_STYLE) as EdgeRouteKind[])
    .map(
      (kind) =>
        `<marker id="${markerId(kind)}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 z" style="fill: ${EDGE_STYLE[kind].color}"></path></marker>`,
    )
    .join("");
  const paths = routes
    .map((r) => {
      const st = EDGE_STYLE[r.kind];
      return `<path d="${r.d}" fill="none" data-edge="${esc(r.from)}->${esc(r.to)}" style="stroke: ${st.color}; stroke-width: ${st.width}${st.dash ? `; stroke-dasharray: ${st.dash}` : ""}" marker-end="url(#${markerId(r.kind)})"></path>`;
    })
    .join("");
  const lanes = routes
    .filter((r) => r.lane)
    .map((r) => `<rect x="${r.lane!.x}" y="${r.lane!.y}" width="${r.lane!.w}" height="${r.lane!.h}" rx="${r.lane!.w / 2}" style="fill: ${EDGE_STYLE.skip.color}; fill-opacity: 0.14"></rect>`)
    .join("");
  svg.innerHTML = `<defs>${defs}</defs>${lanes}${paths}`;
}

function renderScopeBar(current: string, attr: "data-scope-dag" | "data-scope-mermaid"): string {
  const parts = current ? current.split(".") : [];
  let html = `<button class="hover:underline text-[var(--color-accent)] cursor-pointer" ${attr}="">root</button>`;
  let acc = "";
  parts.forEach((p) => {
    acc = acc ? `${acc}.${p}` : p;
    const target = findModuleByPath(rootModule(), acc);
    html += ` <span>/</span> `;
    html +=
      target && hasLocalGraph(target)
        ? `<button class="hover:underline text-[var(--color-accent)] cursor-pointer" ${attr}="${esc(acc)}">${esc(p)}</button>`
        : `<span>${esc(p)}</span>`;
  });
  return `<div class="flex items-center gap-1 flex-wrap text-xs font-mono text-[var(--theme-prose-lead)] mb-3 w-full max-w-xl"><span class="mr-1">Scope:</span>${html}</div>`;
}

export function renderDagView() {
  const container = document.getElementById("dag-container");
  if (!container) return;
  let mod = findModuleByPath(rootModule(), DAG_SCOPE);
  if (!mod || !hasLocalGraph(mod)) {
    const root = rootModule();
    if (root && hasLocalGraph(root)) {
      DAG_SCOPE = "";
      mod = root;
    }
  }
  if (!mod || !hasLocalGraph(mod)) {
    container.innerHTML =
      renderScopeBar(DAG_SCOPE, "data-scope-dag") +
      `<div class="text-center py-12 text-[var(--theme-prose-lead)]">No local graph found for this scope.</div>`;
    return;
  }
  container.innerHTML = renderScopeBar(DAG_SCOPE, "data-scope-dag") + renderLocalPipelineHtml(mod);
  mountFlowEdges(container);
}

// ---------------------------------------------------------------------------
// Hierarchical Tensor Tree View (grouped by owning module scope)
// ---------------------------------------------------------------------------

interface TreeBranch {
  _children: Record<string, TreeBranch>;
  _nodes: NodeData[];
  _paramCount: number;
  _bytes: number;
}

export function renderTree() {
  const filtered = NODES_DATA.filter((item) => {
    const matchesKind = currentKindFilter === "all" || item.kind === currentKindFilter;
    const matchesQuery =
      currentQuery === "" ||
      item.name.toLowerCase().includes(currentQuery) ||
      (item.module || "").toLowerCase().includes(currentQuery) ||
      (item.shape && item.shape.toLowerCase().includes(currentQuery)) ||
      (item.strategy && item.strategy.toLowerCase().includes(currentQuery));
    return matchesKind && matchesQuery;
  });

  const container = document.getElementById("tree-container");
  if (!container) return;

  if (filtered.length === 0) {
    container.innerHTML = `<div class="text-center py-12 text-[var(--theme-prose-lead)] text-sm">No nodes matched the filter criteria.</div>`;
    return;
  }

  const hierarchy = buildHierarchy(filtered);
  let outputHtml = "";
  if (hierarchy._nodes.length > 0) {
    outputHtml += renderBranch("", "(Root Scope)", { ...hierarchy, _children: {} }, true);
  }
  Object.keys(hierarchy._children).forEach((key) => {
    outputHtml += renderBranch(key, key, hierarchy._children[key], false);
  });
  container.innerHTML = outputHtml;
}

function buildHierarchy(data: NodeData[]): TreeBranch {
  const empty = (): TreeBranch => ({ _children: {}, _nodes: [], _paramCount: 0, _bytes: 0 });
  const root = empty();
  data.forEach((item) => {
    const scope = item.module ?? item.name.split(".").slice(0, -1).join(".");
    const parts = scope ? scope.split(".") : [];
    const isParam = item.kind === "Param";
    let curr = root;
    if (parts.length === 0) {
      if (isParam) root._paramCount += item.elements || 0;
      root._bytes += item.bytes || 0;
    }
    for (const p of parts) {
      if (!curr._children[p]) curr._children[p] = empty();
      curr = curr._children[p];
      if (isParam) curr._paramCount += item.elements || 0;
      curr._bytes += item.bytes || 0;
    }
    curr._nodes.push(item);
  });
  return root;
}

function renderBranch(path: string, label: string, node: TreeBranch, open: boolean): string {
  const mod = path ? findModuleByPath(rootModule(), path) : null;
  const typeBadge = mod?.module_type
    ? `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--theme-prose-lead)]">${esc(mod.module_type)}</span>`
    : "";
  const inspect = mod
    ? `<button class="px-2 py-0.5 bg-[var(--bg-secondary)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-color)] rounded text-[10px] font-medium cursor-pointer" data-inspect="${esc(path)}">🔍 Inspect</button>`
    : "";
  let html = `
    <details class="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl overflow-hidden mb-2 group" ${open ? "open" : ""}>
      <summary class="bg-[var(--bg-card)] px-4 py-3 cursor-pointer font-semibold flex items-center justify-between list-none select-none hover:bg-[var(--bg-card-hover)] transition-colors">
        <div class="flex items-center gap-2 font-mono text-xs text-[var(--color-accent)]">
          <span class="transition-transform group-open:rotate-90 inline-block text-[10px]">▶</span>
          <span>${esc(label)}</span>
          ${typeBadge}
        </div>
        <div class="text-[11px] text-[var(--theme-prose-lead)] font-mono flex items-center gap-3">
          ${node._paramCount > 0 ? `<span>⚙️ ${formatNumber(node._paramCount)} params</span>` : ""}
          <span>${formatBytes(node._bytes)}</span>
          ${inspect}
        </div>
      </summary>
      <div class="p-3 flex flex-col gap-2">
        ${renderNodeTable(node._nodes)}
  `;
  Object.keys(node._children).forEach((childKey) => {
    const childPath = path ? `${path}.${childKey}` : childKey;
    html += renderBranch(childPath, childPath, node._children[childKey], false);
  });
  html += `
      </div>
    </details>
  `;
  return html;
}

function renderNodeTable(nodes: NodeData[]): string {
  if (!nodes || nodes.length === 0) return "";
  const tones: Record<NodeKind, string> = {
    Param: "text-blue-500 bg-blue-500/10",
    Input: "text-amber-500 bg-amber-500/10",
    Buffer: "text-slate-500 bg-slate-500/10",
    Activation: "text-purple-400 bg-purple-400/10",
  };
  const badge = (kind: NodeKind) =>
    `<span class="${tones[kind]} px-1.5 py-0.5 rounded text-[10px] font-bold">${esc(kind)}</span>`;
  let html = `
    <table class="w-full text-xs text-left border-collapse mt-1">
      <thead>
        <tr class="border-b border-[var(--border-color)] text-[10.5px] uppercase text-[var(--theme-prose-lead)]">
          <th class="py-1.5 px-2">Tensor Name</th>
          <th class="py-1.5 px-2">Kind</th>
          <th class="py-1.5 px-2">Shape</th>
          <th class="py-1.5 px-2">Elements</th>
          <th class="py-1.5 px-2">Size</th>
        </tr>
      </thead>
      <tbody>
  `;
  nodes.forEach((n) => {
    html += `
      <tr class="border-b border-[var(--border-color)]/50 hover:bg-black/5 dark:hover:bg-white/5 cursor-pointer" data-inspect="${esc(n.name)}">
        <td class="py-1.5 px-2 font-mono font-medium">${esc(n.name)}</td>
        <td class="py-1.5 px-2">${badge(n.kind)}</td>
        <td class="py-1.5 px-2 font-mono text-[11px]">${esc(n.shape)}</td>
        <td class="py-1.5 px-2 font-mono text-[11px]">${formatNumber(n.elements)}</td>
        <td class="py-1.5 px-2 font-mono text-[11px] text-[var(--theme-prose-lead)]">${formatBytes(n.bytes)}</td>
      </tr>
    `;
  });
  html += `</tbody></table>`;
  return html;
}

// ---------------------------------------------------------------------------
// Mermaid local-graph view
// ---------------------------------------------------------------------------

/** Mermaid click argument: `scope:<path>` drills into a child graph, `inspect:<key>` opens the key in the inspector tab. */
function mermaidClickArg(n: FlowNode, drill: boolean): string {
  if (drill && n.kind === "module") {
    const child = findModuleByPath(rootModule(), n.ref);
    if (child && hasLocalGraph(child)) return `scope:${child.path}`;
  }
  return `inspect:${n.ref}`;
}

function renderMermaidInto(mod: ModuleNode, target: HTMLElement, codeEl: HTMLElement | null, drill: boolean) {
  const { code } = buildMermaidLocalGraph(mod, (n) => mermaidClickArg(n, drill));
  if (codeEl) codeEl.textContent = code;

  const renderId = `mermaid-svg-${++mermaidRenderSeq}`;
  mermaidQueue = mermaidQueue.then(async () => {
    const isDark = document.documentElement.classList.contains("dark");
    mermaid.initialize(getMermaidConfig(isDark, { securityLevel: "loose" }));
    try {
      const { svg, bindFunctions } = await mermaid.render(renderId, code);
      target.innerHTML = DOMPurify.sanitize(svg);
      bindFunctions?.(target);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.warn("Mermaid render error:", err);
      target.textContent = `Mermaid render error: ${message}`;
    }
  });
}

export function renderMermaidView() {
  const codeEl = document.getElementById("mermaid-code");
  const target = document.getElementById("mermaid-diagram-target");
  const bar = document.getElementById("mermaid-scope-bar");
  if (!codeEl || !target) return;
  let mod = findModuleByPath(rootModule(), MERMAID_SCOPE);
  if (!mod || !hasLocalGraph(mod)) {
    const root = rootModule();
    if (root && hasLocalGraph(root)) {
      MERMAID_SCOPE = "";
      mod = root;
    }
  }
  if (bar) bar.innerHTML = renderScopeBar(MERMAID_SCOPE, "data-scope-mermaid");
  if (!mod || !hasLocalGraph(mod)) {
    codeEl.textContent = "";
    target.textContent = "No local graph found for this scope.";
    return;
  }
  renderMermaidInto(mod, target, codeEl, true);
}

function onMermaidNodeClick(arg: string) {
  if (arg.startsWith("scope:")) {
    MERMAID_SCOPE = arg.slice("scope:".length);
    renderMermaidView();
  } else if (arg.startsWith("inspect:")) {
    navigateInspector(arg.slice("inspect:".length));
  }
}

// ---------------------------------------------------------------------------
// Inspector & Details
// ---------------------------------------------------------------------------

interface InspectTarget {
  key: string;
  module: ModuleNode | null;
  nodes: NodeData[];
  params: NodeData[];
  ops: OpData[];
  formula: string;
  icon: string;
  /** Module path owning a tensor target (used for breadcrumbs). */
  owner: string;
}

function resolveInspectTarget(rawKey: string): InspectTarget {
  const key = rawKey === "root" ? "" : rawKey;
  const mod = findModuleByPath(rootModule(), key);
  if (mod) {
    const { nodes, params, ops } = collectSubtree(mod);
    return {
      key,
      module: mod,
      nodes,
      params,
      ops,
      formula: mod.formula || FORMULAS_DATA[key] || "\\text{No formula registered for this module}",
      icon: "🧩",
      owner: key,
    };
  }
  const node = NODES_DATA.find((n) => n.name === key);
  const ops = OPS_DATA.filter((o) => o.name === key);
  return {
    key,
    module: null,
    nodes: node ? [node] : [],
    params: node && node.kind === "Param" ? [node] : [],
    ops,
    formula: FORMULAS_DATA[key] || ops.find((o) => o.formula)?.formula || "\\text{No formula registered for this tensor}",
    icon: node ? nodeKindIcon(node.kind) : "📦",
    owner: node?.module ?? ops[0]?.module ?? "",
  };
}

function breadcrumbHtml(t: InspectTarget): string {
  let html = `<button class="hover:underline text-[var(--color-accent)] cursor-pointer" data-inspect="root">root</button>`;
  let acc = "";
  for (const part of t.owner ? t.owner.split(".") : []) {
    acc = acc ? `${acc}.${part}` : part;
    html += ` <span>/</span> <button class="hover:underline text-[var(--color-accent)] cursor-pointer" data-inspect="${esc(acc)}">${esc(part)}</button>`;
  }
  if (!t.module && t.key) html += ` <span>/</span> <span>${esc(t.key.split(".").pop())}</span>`;
  return html;
}

function renderOpList(ops: OpData[]): string {
  if (!ops || ops.length === 0) return `<div class="text-xs text-[var(--theme-prose-lead)]">No ops registered for this node.</div>`;
  let html = `<div class="flex flex-col gap-2">`;
  ops.forEach((o) => {
    html += `
      <div class="flex items-center justify-between bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg p-2.5 font-mono text-xs">
        <div>
          <span class="font-bold text-[var(--color-accent)]">${esc(o.op_type)}</span>: <span>${esc(o.name)}</span>
          ${o.module !== undefined ? `<span class="text-[10px] text-[var(--theme-prose-lead)]"> @ ${esc(o.module || "root")}</span>` : ""}
        </div>
        <div class="text-[11px] text-[var(--theme-prose-lead)]">${esc(o.input_shape || "")} ➔ ${esc(o.output_shape || "")}</div>
      </div>
    `;
  });
  html += `</div>`;
  return html;
}

/** Local data-flow pipeline, Mermaid local graph and direct child cards of a module. */
function renderSubmoduleTreeHtml(mod: ModuleNode | null, mermaidTargetId: string): string {
  if (!mod) return `<div class="text-xs text-[var(--theme-prose-lead)] py-3 text-center">Tensor node (no submodules).</div>`;
  let html = renderLocalPipelineHtml(mod);
  if (hasLocalGraph(mod)) {
    html += `
      <div class="text-xs font-bold uppercase tracking-wider text-[var(--theme-prose-lead)] mb-2">Local Graph (Mermaid)</div>
      <div class="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-3 overflow-x-auto min-h-[120px] flex items-center [&>svg]:mx-auto mb-5" id="${mermaidTargetId}"></div>`;
  } else {
    html += `<div class="text-xs text-[var(--theme-prose-lead)] py-3 text-center">Parameterized leaf module: no internal data-flow graph is exported. See the operations list.</div>`;
  }

  const children = mod.children || [];
  if (children.length === 0) return html;
  html += `<div class="text-xs font-bold uppercase tracking-wider text-[var(--theme-prose-lead)] mb-3">Submodule Components</div>`;
  html += `<div class="grid grid-cols-1 md:grid-cols-2 gap-3.5">`;
  children.forEach((child) => {
    const sub = collectSubtree(child);
    const childParams = child.total_params ?? sub.params.reduce((a, p) => a + (p.elements || 0), 0);
    const childBytes = child.total_bytes ?? sub.params.reduce((a, p) => a + (p.bytes || 0), 0);
    html += `
      <div class="bg-[var(--bg-secondary)] border border-[var(--border-color)] hover:border-[var(--color-accent)] rounded-xl p-3.5 transition-all shadow-xs cursor-pointer group" data-inspect="${esc(child.path)}">
        <div class="flex items-center justify-between pb-2 mb-2 border-b border-[var(--border-color)]">
          <div class="flex items-center gap-2">
            <span class="text-base">🧩</span>
            <span class="font-mono text-xs font-bold text-[var(--color-accent)] group-hover:underline">${esc(child.name)}</span>
          </div>
          <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-[var(--bg-card)] border border-[var(--border-color)] text-[var(--theme-prose-lead)] font-mono">${esc(child.module_type || "Module")}</span>
        </div>
        ${child.formula ? `<div class="bg-[var(--bg-card)] border border-[var(--border-color)]/60 rounded px-2.5 py-1 text-[11px] text-[var(--theme-prose-lead)] mb-2 overflow-x-auto [&_.katex-display]:my-1" title="${esc(child.formula)}">${katexHtml(child.formula)}</div>` : ""}
        <div class="grid grid-cols-2 gap-2 text-[11px] text-[var(--theme-prose-lead)] mb-1">
          <div>⚙️ Params: <span class="font-mono font-medium text-[var(--color-text-primary)]">${formatNumber(childParams)}</span></div>
          <div>⚡ Ops: <span class="font-mono font-medium text-[var(--color-text-primary)]">${sub.ops.length}</span></div>
          <div>📦 Weights: <span class="font-mono font-medium text-[var(--color-text-primary)]">${sub.params.length} tensors</span></div>
          <div>💾 Size: <span class="font-mono font-medium text-[var(--color-text-primary)]">${formatBytes(childBytes)}</span></div>
        </div>
      </div>
    `;
  });
  html += `</div>`;
  return html;
}

function mountMermaid(mod: ModuleNode | null, targetId: string) {
  const el = document.getElementById(targetId);
  if (mod && el && hasLocalGraph(mod)) renderMermaidInto(mod, el, null, false);
}

function renderKatex(formula: string, elId: string) {
  const el = document.getElementById(elId);
  if (!el) return;
  try {
    katex.render(formula, el, { throwOnError: false, displayMode: true });
  } catch (err) {
    console.warn("KaTeX render error:", err);
  }
}

function katexHtml(formula: string): string {
  try {
    return katex.renderToString(formula, { throwOnError: false, displayMode: true });
  } catch (err) {
    console.warn("KaTeX render error:", err);
    return esc(formula);
  }
}

/** Key shown in the inspector tab; `null` until the user picks one (falls back to default_scope). */
let CURRENT_DETAILS_KEY: string | null = null;
let CURRENT_DETAILS_TAB = "structure";
/** Previously inspected targets for the Back button (most recent last). */
const DETAILS_HISTORY: { key: string; tab: string }[] = [];

function isInspectableKey(key: string): boolean {
  return !!findModuleByPath(rootModule(), key) || NODES_DATA.some((n) => n.name === key) || OPS_DATA.some((o) => o.name === key);
}

export function renderDedicatedInspectorView(targetKey?: string, targetTab?: string) {
  if (targetKey !== undefined) CURRENT_DETAILS_KEY = targetKey === "root" ? "" : targetKey;
  if (targetTab) CURRENT_DETAILS_TAB = targetTab;

  const titleEl = document.getElementById("details-title");
  const subEl = document.getElementById("details-subtitle");
  const iconEl = document.getElementById("details-icon");
  const breadcrumbEl = document.getElementById("details-breadcrumb");
  const selectEl = document.getElementById("details-select-module") as HTMLSelectElement | null;
  const backBtn = document.getElementById("details-back") as HTMLButtonElement | null;
  const paramCountEl = document.getElementById("details-tab-param-count");
  const opsCountEl = document.getElementById("details-tab-ops-count");
  const contentEl = document.getElementById("details-content");
  if (!titleEl || !contentEl || !RAW_GRAPH) return;

  if (CURRENT_DETAILS_KEY === null || !isInspectableKey(CURRENT_DETAILS_KEY)) {
    CURRENT_DETAILS_KEY = defaultScope();
  }
  const t = resolveInspectTarget(CURRENT_DETAILS_KEY);

  titleEl.textContent = t.key || "root";
  if (subEl) subEl.textContent = `${t.nodes.length} Nodes · ${t.params.length} Params · ${t.ops.length} Operations`;
  if (iconEl) iconEl.textContent = t.icon;
  if (paramCountEl) paramCountEl.textContent = String(t.params.length);
  if (opsCountEl) opsCountEl.textContent = String(t.ops.length);
  if (breadcrumbEl) breadcrumbEl.innerHTML = breadcrumbHtml(t);
  if (backBtn) backBtn.disabled = DETAILS_HISTORY.length === 0;

  if (selectEl) {
    if (selectEl.options.length === 0) {
      for (const m of listModules(rootModule())) {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = m.path ? `${m.path} (${m.module_type || "Module"})` : "root";
        selectEl.appendChild(opt);
      }
      selectEl.onchange = () => navigateInspector(selectEl.value || "root");
    }
    selectEl.value = t.module ? t.key : t.owner;
  }

  document.querySelectorAll(".details-tab-btn").forEach((btn) => {
    const b = btn as HTMLElement;
    const active = b.dataset.tab === CURRENT_DETAILS_TAB;
    b.classList.toggle("bg-[var(--color-accent)]", active);
    b.classList.toggle("text-white", active);
    b.classList.toggle("text-[var(--theme-prose-lead)]", !active);
    b.onclick = () => renderDedicatedInspectorView(CURRENT_DETAILS_KEY || "root", b.dataset.tab || "structure");
  });

  const card = (title: string, subtitle: string, inner: string) => `
    <div class="bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-5 mb-5 shadow-xs">
      <div class="text-sm font-bold text-[var(--color-accent)] mb-1">${title}</div>
      <div class="text-xs text-[var(--theme-prose-lead)] mb-3">${subtitle}</div>
      ${inner}
    </div>`;

  if (CURRENT_DETAILS_TAB === "structure") {
    if (t.module) {
      contentEl.innerHTML =
        card("Internal Structure", "Scoped local data flow, Mermaid local graph and direct sub-components", renderSubmoduleTreeHtml(t.module, "details-mermaid-target")) +
        card(`Tensor Nodes (${t.nodes.length})`, "Parameters, buffers and activations owned by this module and its descendants", renderNodeTable(t.nodes));
      mountMermaid(t.module, "details-mermaid-target");
      mountFlowEdges(contentEl);
    } else {
      contentEl.innerHTML =
        card("Tensor", `Owned by module <span class="font-mono">${esc(t.owner || "root")}</span>`, renderNodeTable(t.nodes)) +
        card(`Producing Operations (${t.ops.length})`, "Operators that write this tensor during the forward pass", renderOpList(t.ops));
    }
  } else if (CURRENT_DETAILS_TAB === "formula") {
    contentEl.innerHTML = card(
      "Mathematical Formulation",
      "LaTeX definition of the forward transformation",
      `<div id="details-katex-render" class="py-4 px-2 overflow-x-auto text-lg text-center bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl my-2"></div>`,
    );
    renderKatex(t.formula, "details-katex-render");
  } else if (CURRENT_DETAILS_TAB === "params") {
    contentEl.innerHTML = card(`Learnable Parameters (${t.params.length})`, "Weights and biases requiring gradient accumulation", renderNodeTable(t.params));
  } else if (CURRENT_DETAILS_TAB === "ops") {
    contentEl.innerHTML = card(`Registered Operations (${t.ops.length})`, "Operators executed in this module scope during the forward pass", renderOpList(t.ops));
  }
}

// ---------------------------------------------------------------------------
// App Entrypoint & Event Bindings
// ---------------------------------------------------------------------------

declare global {
  interface Window {
    navigateInspector?: (targetKey: string, targetTab?: string) => void;
    onMermaidNodeClick?: (arg: string) => void;
  }
}

/** Show one of the top-level views (`dag`, `tree`, `mermaid`, `details`) without re-rendering it. */
function activateTab(tab: string) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    const active = (b as HTMLElement).dataset.tab === tab;
    b.classList.toggle("active", active);
    b.classList.toggle("bg-[var(--color-accent)]", active);
    b.classList.toggle("text-white", active);
    b.classList.toggle("border-[var(--color-accent)]", active);
    b.classList.toggle("bg-[var(--bg-card)]", !active);
    b.classList.toggle("text-[var(--theme-prose-lead)]", !active);
    b.classList.toggle("border-[var(--border-color)]", !active);
  });
  document.querySelectorAll(".tab-view").forEach((v) => {
    const active = v.id === `view-${tab}`;
    v.classList.toggle("block", active);
    v.classList.toggle("hidden", !active);
  });
}

/**
 * Open a module or tensor in the inspector tab. Every "inspect" click in the page goes through
 * here, so inspection always happens in place instead of in an overlay.
 */
function navigateInspector(targetKey: string, targetTab?: string) {
  const key = targetKey === "root" ? "" : targetKey;
  if (CURRENT_DETAILS_KEY !== null && CURRENT_DETAILS_KEY !== key) {
    DETAILS_HISTORY.push({ key: CURRENT_DETAILS_KEY, tab: CURRENT_DETAILS_TAB });
  }
  activateTab("details");
  renderDedicatedInspectorView(targetKey, targetTab);
  const view = document.getElementById("view-details");
  if (view && view.getBoundingClientRect().top < 0) view.scrollIntoView({ block: "start" });
}

function inspectorBack() {
  const prev = DETAILS_HISTORY.pop();
  if (!prev) return;
  renderDedicatedInspectorView(prev.key || "root", prev.tab);
}

/** Temporarily replace a button label to acknowledge an action in place. */
function flashButton(btn: HTMLElement | null, text: string) {
  if (!btn) return;
  const original = btn.dataset.label ?? btn.textContent ?? "";
  btn.dataset.label = original;
  btn.textContent = text;
  window.setTimeout(() => {
    btn.textContent = original;
  }, 1500);
}

function setLoadStatus(message: string) {
  const el = document.getElementById("load-status");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("hidden", message === "");
}

let documentHandlersBound = false;

function bindDocumentHandlers() {
  if (documentHandlersBound) return;
  documentHandlersBound = true;

  // Delegated navigation: data-* attributes carry canonical keys, so no model string is
  // ever spliced into inline JavaScript.
  document.addEventListener("click", (ev) => {
    const el = (ev.target as HTMLElement | null)?.closest<HTMLElement>("[data-inspect],[data-scope-dag],[data-scope-mermaid]");
    if (!el) return;
    ev.preventDefault();
    ev.stopPropagation();
    if (el.dataset.scopeDag !== undefined) {
      DAG_SCOPE = el.dataset.scopeDag;
      renderDagView();
    } else if (el.dataset.scopeMermaid !== undefined) {
      MERMAID_SCOPE = el.dataset.scopeMermaid;
      renderMermaidView();
    } else if (el.dataset.inspect !== undefined) {
      navigateInspector(el.dataset.inspect || "root");
    }
  });
}

function bindOnce(el: Element | null, event: string, handler: (ev: Event) => void) {
  if (!el) return;
  const h = el as HTMLElement;
  const flag = `bound${event.replace(/[^a-z]/gi, "")}`;
  if (h.dataset[flag]) return;
  h.dataset[flag] = "1";
  h.addEventListener(event, handler);
}

function resolveModelUrl(name: string): string {
  if (name === "sample") return "/tools/visualizer/sample_model_graph.json";
  if (name === "minimal") return "/tools/visualizer/minimal_model_graph.json";
  return `/tools/visualizer/models/${name}.json`;
}

export function initVisualizer(preloadedGraph?: ModelHierarchyGraph) {
  const pageRoot = document.querySelector<HTMLElement>(".visualizer-page");
  if (pageRoot) {
    if (pageRoot.dataset.visualizerInitialized === "1") return;
    pageRoot.dataset.visualizerInitialized = "1";
  }
  window.navigateInspector = navigateInspector;
  window.onMermaidNodeClick = onMermaidNodeClick;
  bindDocumentHandlers();

  const params = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
  const modelParam = params?.get("model");
  const urlParam = params?.get("url");
  const tabParam = params?.get("tab");
  const scopeParam = params?.get("scope");

  const presetSelect = document.getElementById("model-preset-select") as HTMLSelectElement | null;
  if (presetSelect && modelParam) {
    if (Array.from(presetSelect.options).some((o) => o.value === modelParam)) {
      presetSelect.value = modelParam;
    }
  }

  async function fetchAndLoad(targetUrl: string, sourceName: string) {
    setLoadStatus(`Loading ${sourceName}...`);
    try {
      const res = await fetch(targetUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status} (${res.statusText})`);
      const parsed = (await res.json()) as unknown;
      const errors = validateJson(MODEL_GRAPH_SCHEMA, parsed, 4);
      if (errors.length > 0) {
        const details = errors.map((e) => `${e.path}: ${e.message}`).join("; ");
        setLoadStatus(`${sourceName} does not match model_graph.schema.json (schema 2.0): ${details}`);
        return;
      }
      setLoadStatus("");
      loadGraphData(parsed as ModelHierarchyGraph, sourceName);
      if (scopeParam) {
        DAG_SCOPE = scopeParam;
        MERMAID_SCOPE = scopeParam;
        renderDagView();
        renderMermaidView();
      }
      if (tabParam) {
        activateTab(tabParam);
        if (tabParam === "mermaid") renderMermaidView();
        if (tabParam === "dag") renderDagView();
        if (tabParam === "tree") renderTree();
        if (tabParam === "details") renderDedicatedInspectorView();
      }
    } catch (err) {
      setLoadStatus(`Failed to load ${sourceName}: ${(err as Error).message}`);
    }
  }

  if (presetSelect) {
    bindOnce(presetSelect, "change", () => {
      const selected = presetSelect.value;
      if (!selected) return;
      const targetUrl = resolveModelUrl(selected);
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set("model", selected);
      newUrl.searchParams.delete("url");
      window.history.pushState({}, "", newUrl.toString());
      fetchAndLoad(targetUrl, selected);
    });
  }

  // Initial model graph load
  const initialTarget = urlParam || (modelParam ? resolveModelUrl(modelParam) : null);
  if (initialTarget) {
    fetchAndLoad(initialTarget, modelParam || urlParam || "model");
  } else if (preloadedGraph) {
    loadGraphData(preloadedGraph, "bundled sample");
    if (scopeParam) {
      DAG_SCOPE = scopeParam;
      MERMAID_SCOPE = scopeParam;
      renderDagView();
      renderMermaidView();
    }
    if (tabParam) {
      activateTab(tabParam);
      if (tabParam === "mermaid") renderMermaidView();
      if (tabParam === "dag") renderDagView();
      if (tabParam === "tree") renderTree();
      if (tabParam === "details") renderDedicatedInspectorView();
    }
  } else if (!RAW_GRAPH) {
    fetchAndLoad("/tools/visualizer/sample_model_graph.json", "bundled sample");
  }

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    bindOnce(btn, "click", () => {
      const tab = (btn as HTMLElement).dataset.tab;
      if (!tab) return;
      activateTab(tab);
      if (tab === "mermaid") renderMermaidView();
      if (tab === "dag") renderDagView();
      if (tab === "tree") renderTree();
      if (tab === "details") renderDedicatedInspectorView();
    });
  });
  bindOnce(document.getElementById("details-back"), "click", inspectorBack);

  // Filter Buttons
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    bindOnce(btn, "click", () => {
      currentKindFilter = (btn as HTMLElement).dataset.filter || "all";
      document.querySelectorAll(".filter-btn").forEach((b) => {
        b.classList.remove("active", "bg-[var(--color-accent)]", "text-white", "border-[var(--color-accent)]");
        b.classList.add("bg-[var(--bg-card)]", "border-[var(--border-color)]");
      });
      btn.classList.add("active", "bg-[var(--color-accent)]", "text-white", "border-[var(--color-accent)]");
      btn.classList.remove("bg-[var(--bg-card)]", "border-[var(--border-color)]");
      renderTree();
    });
  });

  // Tree Controls
  const searchInput = document.getElementById("search-input") as HTMLInputElement | null;
  bindOnce(searchInput, "input", () => {
    currentQuery = (searchInput?.value || "").trim().toLowerCase();
    renderTree();
  });
  bindOnce(document.getElementById("btn-expand-all"), "click", () => {
    document.querySelectorAll("#tree-container details").forEach((d) => ((d as HTMLDetailsElement).open = true));
  });
  bindOnce(document.getElementById("btn-collapse-all"), "click", () => {
    document.querySelectorAll("#tree-container details").forEach((d) => ((d as HTMLDetailsElement).open = false));
  });

  // Copy buttons (the button is captured before the async write: ev.currentTarget is null afterwards)
  const bindCopy = (buttonId: string, codeId: string, wrap: (code: string) => string = (c) => c) => {
    const button = document.getElementById(buttonId);
    bindOnce(button, "click", (ev) => {
      ev.preventDefault();
      const code = document.getElementById(codeId)?.textContent || "";
      navigator.clipboard.writeText(wrap(code)).then(() => flashButton(button, "✓ Copied"));
    });
  };
  bindCopy("btn-copy-mermaid", "mermaid-code", (code) => "```mermaid\n" + code + "```");
  bindCopy("btn-copy-template", "schema-template-code");
  bindCopy("btn-copy-schema", "schema-json-code");

  // File Upload Handler
  const fileInput = document.getElementById("graph-file-input") as HTMLInputElement | null;
  bindOnce(fileInput, "change", () => {
    const file = fileInput?.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const parsed = JSON.parse(evt.target?.result as string) as unknown;
        const errors = validateJson(MODEL_GRAPH_SCHEMA, parsed, 4);
        if (errors.length > 0) {
          const details = errors.map((e) => `${e.path}: ${e.message}`).join("; ");
          setLoadStatus(`${file.name} does not match model_graph.schema.json (schema 2.0): ${details}`);
          return;
        }
        if (presetSelect) presetSelect.selectedIndex = -1;
        loadGraphData(parsed as ModelHierarchyGraph, file.name);
      } catch (err) {
        setLoadStatus(`${file.name}: failed to parse JSON (${(err as Error).message}).`);
      }
    };
    reader.readAsText(file);
  });
}

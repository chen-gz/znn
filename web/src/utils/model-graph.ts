/**
 * src/utils/model-graph.ts
 *
 * DOM-free data layer for the ZNN model visualizer (ModelHierarchyGraph schema 2.0).
 * Every module carries its own scoped local graph (`ports`, `flow_nodes`, `edges`);
 * this module turns it into an indexed DAG, resolves local ids to canonical paths,
 * detects residual branches by reachability and lays the graph out into rows.
 * Specification: doc/visualization-model-edge-design.md §3.
 */

// Value lists of the enum fields, in schema order (checked against model_graph.schema.json in tests).
export const FLOW_NODE_KINDS = ["port_in", "port_out", "module", "op", "buffer"] as const;
export const EDGE_KINDS = ["data", "buffer"] as const;
export const NODE_KINDS = ["Param", "Input", "Buffer", "Activation"] as const;
export const NODE_STATUSES = ["CUSTOM_INIT", "AUTO_GRAPH", "INPUT", "BUFFER", "OP_OUTPUT"] as const;

export const CANONICAL_MODEL_NAMES = [
  "attention",
  "conv2d",
  "deepseek_moe",
  "embedding",
  "gan_discriminator",
  "gan_generator",
  "gpt",
  "gru",
  "layernorm",
  "linear",
  "lora_linear",
  "lstm",
  "mla",
  "mlp",
  "rnn",
  "stacked_lstm",
  "swiglu",
  "transformer_block",
] as const;

export type FlowNodeKind = (typeof FLOW_NODE_KINDS)[number];
export type EdgeKind = (typeof EDGE_KINDS)[number];
export type NodeKind = (typeof NODE_KINDS)[number];
export type NodeStatus = (typeof NODE_STATUSES)[number];

export interface FlowNode {
  id: string;
  kind: FlowNodeKind;
  ref: string;
  op_type?: string;
  module_type?: string;
  shape?: string;
}

export interface PortEntry {
  id: string;
  ref: string;
  shape?: string;
}

export interface ModulePorts {
  inputs: PortEntry[];
  outputs: PortEntry[];
}

export interface EdgeData {
  from: string;
  to: string;
  shape: string;
  dst_shape?: string;
  transforms?: string[];
  is_skip: boolean;
  kind: EdgeKind;
}

export interface NodeData {
  name: string;
  kind: NodeKind;
  module?: string;
  shape: string;
  elements: number;
  bytes: number;
  status: NodeStatus;
  act?: string;
  strategy?: string;
}

export interface OpData {
  op_type: string;
  name: string;
  module?: string;
  input_shape?: string;
  param_shape?: string;
  output_shape?: string;
  formula?: string | null;
  elements?: number;
  bytes?: number;
}

export interface ModuleNode {
  name: string;
  path: string;
  kind?: "module";
  module_type?: string;
  formula?: string | null;
  total_params?: number;
  total_bytes?: number;
  param_count?: number;
  node_count?: number;
  children?: ModuleNode[];
  parameters?: NodeData[];
  ops?: OpData[];
  ports?: ModulePorts;
  flow_nodes?: FlowNode[];
  edges?: EdgeData[];
  nodes?: NodeData[];
}

export interface GraphSummary {
  total_params?: number;
  total_bytes?: number;
  param_nodes?: number;
  input_nodes?: number;
  buffer_nodes?: number;
  activation_nodes?: number;
  custom_init_count?: number;
  auto_graph_count?: number;
  total_nodes?: number;
}

export interface ModelHierarchyGraph {
  version?: string;
  summary?: GraphSummary;
  default_scope?: string;
  root: ModuleNode;
}

// ---------------------------------------------------------------------------
// Module tree helpers
// ---------------------------------------------------------------------------

export function joinPath(scope: string, id: string): string {
  return scope ? `${scope}.${id}` : id;
}

/** Finds a module by its full path. `""` and `"root"` both address the root module. */
export function findModuleByPath(root: ModuleNode | null | undefined, targetPath: string | null | undefined): ModuleNode | null {
  if (!root || typeof targetPath !== "string") return null;
  const path = targetPath === "root" ? "" : targetPath;
  if ((root.path || "") === path) return root;
  if (path !== "" && root.path && !path.startsWith(root.path + ".")) return null;
  for (const child of root.children || []) {
    const res = findModuleByPath(child, path);
    if (res) return res;
  }
  return null;
}

/** Depth-first list of all modules in the tree (root first). */
export function listModules(root: ModuleNode | null | undefined): ModuleNode[] {
  const out: ModuleNode[] = [];
  const walk = (m: ModuleNode) => {
    out.push(m);
    (m.children || []).forEach(walk);
  };
  if (root) walk(root);
  return out;
}

/** Collects the nodes, ops and parameters of a module and all of its descendants. */
export function collectSubtree(mod: ModuleNode) {
  const nodes: NodeData[] = [];
  const ops: OpData[] = [];
  const seen = new Set<string>();
  const walk = (m: ModuleNode) => {
    for (const n of m.nodes || []) {
      if (!seen.has(n.name)) {
        seen.add(n.name);
        nodes.push(n);
      }
    }
    for (const p of m.parameters || []) {
      if (!seen.has(p.name)) {
        seen.add(p.name);
        nodes.push({ ...p, kind: "Param", module: m.path });
      }
    }
    ops.push(...(m.ops || []));
    (m.children || []).forEach(walk);
  };
  walk(mod);
  const params = nodes.filter((n) => n.kind === "Param");
  return { nodes, ops, params };
}

// ---------------------------------------------------------------------------
// Scoped local graph
// ---------------------------------------------------------------------------

export interface LocalGraph {
  scope: string;
  nodes: Map<string, FlowNode>;
  /** Topological order of node ids (stable with respect to `flow_nodes`). */
  order: string[];
  edges: EdgeData[];
  incoming: Map<string, EdgeData[]>;
  outgoing: Map<string, EdgeData[]>;
}

function inferKind(id: string): FlowNodeKind {
  if (id.startsWith("@in")) return "port_in";
  if (id.startsWith("@out")) return "port_out";
  return "op";
}

export function hasLocalGraph(mod: ModuleNode | null | undefined): boolean {
  return !!mod && !!mod.edges && mod.edges.length > 0;
}

export function buildLocalGraph(mod: ModuleNode): LocalGraph {
  const scope = mod.path || "";
  const nodes = new Map<string, FlowNode>();
  for (const n of mod.flow_nodes || []) nodes.set(n.id, n);

  const edges = mod.edges || [];
  const incoming = new Map<string, EdgeData[]>();
  const outgoing = new Map<string, EdgeData[]>();
  for (const e of edges) {
    for (const id of [e.from, e.to]) {
      if (!nodes.has(id)) nodes.set(id, { id, kind: inferKind(id), ref: joinPath(scope, id) });
    }
    if (!incoming.has(e.to)) incoming.set(e.to, []);
    incoming.get(e.to)!.push(e);
    if (!outgoing.has(e.from)) outgoing.set(e.from, []);
    outgoing.get(e.from)!.push(e);
  }

  // Kahn's algorithm; ties are broken by declaration order so the layout is deterministic.
  const ids = Array.from(nodes.keys());
  const rank = new Map(ids.map((id, i) => [id, i]));
  const indeg = new Map(ids.map((id) => [id, (incoming.get(id) || []).length]));
  const ready = ids.filter((id) => indeg.get(id) === 0);
  const order: string[] = [];
  while (ready.length > 0) {
    ready.sort((a, b) => rank.get(a)! - rank.get(b)!);
    const u = ready.shift()!;
    order.push(u);
    for (const e of outgoing.get(u) || []) {
      const d = indeg.get(e.to)! - 1;
      indeg.set(e.to, d);
      if (d === 0) ready.push(e.to);
    }
  }
  // A malformed (cyclic) graph still renders every node.
  for (const id of ids) if (!order.includes(id)) order.push(id);

  return { scope, nodes, order, edges, incoming, outgoing };
}

/**
 * Resolves a local flow-node id to the canonical global path used by the inspector:
 * ports resolve to their external `ref`, children / ops / buffers to their own `ref`.
 */
export function resolveCanonical(mod: ModuleNode, id: string): string {
  const scope = mod.path || "";
  const flow = (mod.flow_nodes || []).find((n) => n.id === id);
  if (flow && flow.ref) return flow.ref;
  const port = [...(mod.ports?.inputs || []), ...(mod.ports?.outputs || [])].find((p) => p.id === id);
  if (port) return port.ref;
  return joinPath(scope, id);
}

export function isAddNode(node: FlowNode | undefined): boolean {
  return !!node && node.kind === "op" && node.op_type === "Add";
}

function walk(
  start: string,
  next: Map<string, EdgeData[]>,
  pick: (e: EdgeData) => string,
  allowed?: Set<string>,
): Set<string> {
  const seen = new Set<string>();
  const stack = [start];
  while (stack.length > 0) {
    const u = stack.pop()!;
    for (const e of next.get(u) || []) {
      const v = pick(e);
      if (allowed && !allowed.has(v)) continue;
      if (!seen.has(v)) {
        seen.add(v);
        stack.push(v);
      }
    }
  }
  return seen;
}

/** Nodes reachable from `start` (excluding `start`), optionally restricted to `allowed`. */
export function reachableFrom(g: LocalGraph, start: string, allowed?: Set<string>): Set<string> {
  return walk(start, g.outgoing, (e) => e.to, allowed);
}

/** Nodes that can reach `target` (excluding `target`), optionally restricted to `allowed`. */
export function reachableTo(g: LocalGraph, target: string, allowed?: Set<string>): Set<string> {
  return walk(target, g.incoming, (e) => e.from, allowed);
}

/**
 * Transform branch of a residual edge (s -> t): every node on a path from s to a
 * non-skip predecessor w of t, i.e. Reach+(s) ∩ (Reach-(w) ∪ {w}), in topological order.
 */
export function residualBranch(g: LocalGraph, skip: EdgeData, allowed?: Set<string>): string[] {
  const s = skip.from;
  const t = skip.to;
  const fwd = reachableFrom(g, s, allowed);
  const branch = new Set<string>();
  for (const e of g.incoming.get(t) || []) {
    if (e === skip || e.is_skip || e.from === s) continue;
    if (allowed && !allowed.has(e.from)) continue;
    const back = reachableTo(g, e.from, allowed);
    back.add(e.from);
    for (const v of back) if (fwd.has(v)) branch.add(v);
  }
  branch.delete(s);
  branch.delete(t);
  return g.order.filter((id) => branch.has(id));
}

// ---------------------------------------------------------------------------
// Row layout for the card pipeline
// ---------------------------------------------------------------------------

export type PipelineUnit =
  | { type: "node"; id: string }
  | { type: "residual"; skip: EdgeData; target: string; branchIds: string[]; branch: PipelineRow[] };

export interface PipelineRow {
  units: PipelineUnit[];
}

/**
 * Lays a local graph (or the subset `subset` of it) out into rows by longest path.
 * Each residual (skip edge plus its transform branch and the Add target) becomes one
 * unit whose branch is laid out recursively, so nested residuals are handled naturally.
 * Shared sources appear once; their fan-out is expressed by the rows below them.
 */
export function layoutPipeline(g: LocalGraph, subset?: Set<string>): PipelineRow[] {
  const allowed = subset ?? new Set(g.order);
  const ids = g.order.filter((id) => allowed.has(id));
  const rank = new Map(ids.map((id, i) => [id, i]));

  // 1. Residual blocks: outermost first, only disjoint blocks are claimed at this level.
  interface Block {
    skip: EdgeData;
    target: string;
    branchIds: string[];
    members: Set<string>;
  }
  const candidates: Block[] = [];
  for (const e of g.edges) {
    if (!e.is_skip || !allowed.has(e.from) || !allowed.has(e.to)) continue;
    if (candidates.some((b) => b.target === e.to)) continue;
    const branchIds = residualBranch(g, e, allowed);
    if (branchIds.length === 0) continue;
    candidates.push({ skip: e, target: e.to, branchIds, members: new Set([...branchIds, e.to]) });
  }
  candidates.sort((a, b) => b.members.size - a.members.size);
  const unitOf = new Map<string, number>();
  const blocks: Block[] = [];
  for (const b of candidates) {
    if (b.members.has(b.skip.from)) continue;
    if ([...b.members].some((id) => unitOf.has(id))) continue;
    const idx = ids.length + blocks.length;
    blocks.push(b);
    b.members.forEach((id) => unitOf.set(id, idx));
  }
  ids.forEach((id, i) => {
    if (!unitOf.has(id)) unitOf.set(id, i);
  });

  // 2. Unit graph and longest-path levels.
  const unitIds = Array.from(new Set(ids.map((id) => unitOf.get(id)!)));
  const unitRank = new Map<number, number>();
  for (const id of ids) {
    const u = unitOf.get(id)!;
    if (!unitRank.has(u)) unitRank.set(u, rank.get(id)!);
  }
  const succ = new Map<number, Set<number>>(unitIds.map((u) => [u, new Set<number>()]));
  const indeg = new Map<number, number>(unitIds.map((u) => [u, 0]));
  for (const e of g.edges) {
    if (!allowed.has(e.from) || !allowed.has(e.to)) continue;
    const a = unitOf.get(e.from)!;
    const b = unitOf.get(e.to)!;
    if (a === b || succ.get(a)!.has(b)) continue;
    succ.get(a)!.add(b);
    indeg.set(b, indeg.get(b)! + 1);
  }
  const level = new Map<number, number>(unitIds.map((u) => [u, 0]));
  const ready = unitIds.filter((u) => indeg.get(u) === 0);
  const visited = new Set<number>();
  while (ready.length > 0) {
    const u = ready.shift()!;
    visited.add(u);
    for (const v of succ.get(u)!) {
      level.set(v, Math.max(level.get(v)!, level.get(u)! + 1));
      indeg.set(v, indeg.get(v)! - 1);
      if (indeg.get(v) === 0) ready.push(v);
    }
  }
  const pred = new Map<number, number[]>(unitIds.map((u) => [u, []]));
  for (const [a, targets] of succ) for (const b of targets) pred.get(b)!.push(a);
  // Sources (input ports, buffers) sit directly above their first consumer instead of all
  // piling up in the first row, which keeps their edges short and the first row narrow.
  for (const u of unitIds) {
    if (!visited.has(u) || pred.get(u)!.length > 0 || succ.get(u)!.size === 0) continue;
    const firstUse = Math.min(...Array.from(succ.get(u)!, (v) => level.get(v)!));
    level.set(u, Math.max(level.get(u)!, firstUse - 1));
  }

  // 3. Rows, ordered top-down by the barycenter of each unit's predecessors (normalized
  //    position in their rows), so edges drawn between cards cross as little as possible.
  const byLevel = new Map<number, number[]>();
  for (const u of unitIds) {
    const l = visited.has(u) ? level.get(u)! : unitIds.length;
    if (!byLevel.has(l)) byLevel.set(l, []);
    byLevel.get(l)!.push(u);
  }
  const position = new Map<number, number>();
  const levels = Array.from(byLevel.keys()).sort((a, b) => a - b);
  for (const l of levels) {
    const row = byLevel.get(l)!;
    const bary = new Map<number, number>();
    for (const u of row) {
      const ps = pred.get(u)!.filter((p) => position.has(p));
      if (ps.length > 0) bary.set(u, ps.reduce((acc, p) => acc + position.get(p)!, 0) / ps.length);
    }
    row.sort((a, b) => {
      const ba = bary.get(a);
      const bb = bary.get(b);
      // Units fed from earlier rows follow their predecessors; sources pulled down into this
      // row (no placed predecessor) go to the right, in topological order.
      if (ba !== undefined && bb !== undefined && ba !== bb) return ba - bb;
      if ((ba === undefined) !== (bb === undefined)) return ba === undefined ? 1 : -1;
      return unitRank.get(a)! - unitRank.get(b)!;
    });
    row.forEach((u, i) => position.set(u, (i + 0.5) / row.length));
  }
  return levels.map((l) => ({
    units: byLevel.get(l)!.map((u): PipelineUnit => {
      if (u < ids.length) return { type: "node", id: ids[u] };
      const b = blocks[u - ids.length];
      return {
        type: "residual",
        skip: b.skip,
        target: b.target,
        branchIds: b.branchIds,
        branch: layoutPipeline(g, new Set(b.branchIds)),
      };
    }),
  }));
}

/** Minimum width of one node card in the card pipeline (rem). */
export const PIPELINE_CARD_MIN_REM = 6.75;
/** Horizontal gap between the units of one row (rem, matches `gap-x-2`). */
export const PIPELINE_GAP_REM = 0.5;
/** Horizontal chrome of a residual unit: frame `pl-7 pr-3`, inner box `p-2.5` and borders (rem). */
export const PIPELINE_RESIDUAL_CHROME_REM = 4;

/** Minimum width (rem) a pipeline unit needs so none of its cards is squeezed below the card minimum. */
export function pipelineUnitMinWidthRem(unit: PipelineUnit): number {
  return unit.type === "node" ? PIPELINE_CARD_MIN_REM : pipelineMinWidthRem(unit.branch) + PIPELINE_RESIDUAL_CHROME_REM;
}

/** Minimum width (rem) of a laid-out pipeline: the widest row, including gaps and nested residual frames. */
export function pipelineMinWidthRem(rows: PipelineRow[]): number {
  let width = PIPELINE_CARD_MIN_REM;
  for (const row of rows) {
    const units = row.units.map(pipelineUnitMinWidthRem);
    const rowWidth = units.reduce((sum, w) => sum + w, 0) + PIPELINE_GAP_REM * Math.max(0, units.length - 1);
    width = Math.max(width, rowWidth);
  }
  return width;
}

// ---------------------------------------------------------------------------
// Labels and Mermaid code generation
// ---------------------------------------------------------------------------

export function edgeShapeLabel(e: EdgeData): string {
  if (e.dst_shape && e.dst_shape !== e.shape) return `${e.shape} → ${e.dst_shape}`;
  return e.shape || "";
}

export interface MermaidLocalGraph {
  code: string;
  /** Mermaid node id -> flow node (for click handling). */
  nodeIds: Map<string, FlowNode>;
}

function mermaidText(s: string): string {
  return s.replace(/"/g, "'").replace(/[<>]/g, "");
}

/**
 * Mermaid flowchart of one module's local graph. Node shapes come only from
 * `kind` / `op_type`; skip edges are dashed, buffer edges dotted, and edge labels
 * show `shape → dst_shape`. `clickCall(node)` returns the argument for the click callback.
 */
export function buildMermaidLocalGraph(
  mod: ModuleNode,
  clickCall?: (node: FlowNode) => string | null,
): MermaidLocalGraph {
  const g = buildLocalGraph(mod);
  const nodeIds = new Map<string, FlowNode>();
  const safe = new Map<string, string>();
  let code = "flowchart TD\n";

  g.order.forEach((id, i) => {
    const n = g.nodes.get(id)!;
    const sid = `n${i}`;
    safe.set(id, sid);
    nodeIds.set(sid, n);
    const label = mermaidText(id);
    switch (n.kind) {
      case "port_in":
        code += `  ${sid}(["📥 ${label} · ${mermaidText(n.ref)}"])\n`;
        break;
      case "port_out":
        code += `  ${sid}(["📤 ${label} · ${mermaidText(n.ref)}"])\n`;
        break;
      case "buffer":
        code += `  ${sid}[("🧱 ${label}")]\n`;
        break;
      case "module":
        code += `  ${sid}["🧩 ${label} : ${mermaidText(n.module_type || "Module")}"]\n`;
        break;
      default:
        code += isAddNode(n)
          ? `  ${sid}(("⊕ ${label}"))\n`
          : `  ${sid}("⚡ ${label} : ${mermaidText(n.op_type || "Op")}")\n`;
    }
  });

  const linkStyles: string[] = [];
  g.edges.forEach((e, i) => {
    const a = safe.get(e.from)!;
    const b = safe.get(e.to)!;
    let label = mermaidText(edgeShapeLabel(e));
    if (e.is_skip) label = label ? `skip · ${label}` : "skip";
    const arrow = e.is_skip || e.kind === "buffer" ? "-.->" : "-->";
    code += label ? `  ${a} ${arrow}|"${label}"| ${b}\n` : `  ${a} ${arrow} ${b}\n`;
    if (e.is_skip) linkStyles.push(`  linkStyle ${i} stroke:#f59e0b,stroke-width:2px,stroke-dasharray:8 4\n`);
    else if (e.kind === "buffer") linkStyles.push(`  linkStyle ${i} stroke:#94a3b8,stroke-dasharray:2 3\n`);
  });
  code += linkStyles.join("");

  if (clickCall) {
    for (const [sid, n] of nodeIds) {
      const arg = clickCall(n);
      if (arg !== null) code += `  click ${sid} call onMermaidNodeClick("${mermaidText(arg)}")\n`;
    }
  }
  return { code, nodeIds };
}

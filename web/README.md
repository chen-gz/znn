# ZNN Model Visualizer (`web/`)

Interactive web frontend for inspecting `ModelHierarchyGraph` (Schema 2.0) JSON files exported by `znn`.

## Overview

- **Zero Data Duplication**: Directly reads `../src/nn/model_graph.schema.json`, `../examples/sample_model_graph.json`, `../examples/minimal_model_graph.json`, and `../examples/models/*.json`.
- **Architecture & Edge Specification**: [`../doc/visualization-model-edge-design.md`](../doc/visualization-model-edge-design.md)
- **Modules**:
  - `src/utils/model-graph.ts` — Scoped local DAG extraction, canonical path resolution, residual branch detection, and layered pipeline layout.
  - `src/utils/flow-edges.ts` — DOM-independent orthogonal and residual skip edge routing.
  - `src/utils/json-schema.ts` — Draft 2020-12 JSON Schema validator and field table generator.
  - `src/scripts/visualizer.ts` — Interactive DAG pipeline, hierarchical module tree, Mermaid flowchart, and in-page module inspector with KaTeX formula rendering.

## Commands

```bash
cd web
npm install
npm test            # Run Vitest unit tests (model-graph, flow-edges, json-schema, visualizer)
npm run typecheck   # TypeScript strict check
npm run dev         # Start local Vite dev server
npm run build       # Build static bundle into web/dist/
```

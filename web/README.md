# ZNN Model Architecture & Computation Graph Explorer (Web Frontend)

This is the standalone web frontend subproject for visual inspection of `znn` autodiff graphs and model hierarchies.

## Features
- **Pure Client-Side**: No Node.js / npm build step required. Runs directly in any modern browser via standard HTML5, CSS3, and ES6 JavaScript.
- **JSON Input Driven**: Accepts `ModelHierarchyGraph` JSON exported by `znn` via file picker, drag-and-drop, or embedded data tag.
- **Three Core Views**:
  1. **Architecture & Skip Connections**: Dynamically ordered topological stages with residual shortcut branches and parallel inputs.
  2. **Hierarchical Module Tree**: Collapsible, searchable module tree showing parameters, memory footprints, and sublayers.
  3. **Mermaid Flowchart**: Interactive SVG graph rendered from topological DAG edges, featuring **clickable nodes** that open the detailed Node Inspector.
- **LaTeX Math Formulas**: Integrated with KaTeX to render LaTeX equations automatically generated and inferred from `autodiff.OpType` and `nn` modules.
- **Node Inspector**: Modal drawer detailing inbound connections, downstream flows, operation types, tensor shapes, and trainable parameters.

## Directory Structure
```
web/
├── index.html        # Main application entry point
├── css/
│   └── style.css     # Modular dark-themed UI styling
├── js/
│   └── app.js        # Core rendering, topological sort, and event handling logic
└── README.md
```

## How to Run

### Option 1: Direct Browser Access
Simply open `web/index.html` in your browser:
```bash
open web/index.html
```
Click **"Choose JSON File"** to load any exported model graph JSON (such as `examples/sample_model_graph.json`).

### Option 2: Local HTTP Server
Start a lightweight Python HTTP server from the project root:
```bash
python3 -m http.server 8000
```
Navigate to:
```
http://localhost:8000/web/
```

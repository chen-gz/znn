/**
 * ZNN Visualizer - Frontend Application Logic
 * Pure client-side renderer accepting ModelHierarchyGraph JSON.
 */

// Global State
let RAW_GRAPH = null;
let RAW_DATA = null;
let NODES_DATA = [];
let EDGES_DATA = [];
let OPS_DATA = [];
let FORMULAS_DATA = {};
let currentKindFilter = 'all';
let currentQuery = '';

// ---------------------------------------------------------------------------
// 1. Data Initialization & File Loading
// ---------------------------------------------------------------------------

function collectAllFromTree(root) {
  const nodes = [];
  const edges = [];
  const ops = [];
  const formulas = {};

  function traverse(m) {
    if (!m) return;
    if (m.formula && m.path) formulas[m.path] = m.formula;
    if (m.nodes) nodes.push(...m.nodes);
    if (m.edges) edges.push(...m.edges);
    if (m.ops) {
      ops.push(...m.ops);
      for (const op of m.ops) {
        if (op.formula && op.name) formulas[op.name] = op.formula;
        if (op.formula && op.op_type && !formulas[op.op_type]) formulas[op.op_type] = op.formula;
      }
    }
    if (m.children) {
      for (const child of m.children) traverse(child);
    }
  }

  traverse(root);
  return { nodes, edges, ops, formulas };
}

function loadGraphData(graphObj) {
  if (!graphObj) return;
  RAW_GRAPH = graphObj;
  const treeData = collectAllFromTree(graphObj.root);
  NODES_DATA = (graphObj.nodes && graphObj.nodes.length > 0) ? graphObj.nodes : treeData.nodes;
  EDGES_DATA = (graphObj.edges && graphObj.edges.length > 0) ? graphObj.edges : treeData.edges;
  OPS_DATA = (graphObj.ops && graphObj.ops.length > 0) ? graphObj.ops : treeData.ops;
  FORMULAS_DATA = Object.assign({}, treeData.formulas, graphObj.formulas || {});

  // Update Model Name in Title
  const modelName = graphObj.model_name || 'Neural Network';
  const titleEl = document.getElementById('report-model-name');
  if (titleEl) {
    titleEl.textContent = `⚡ ZNN Architecture: ${modelName}`;
  }

  initKpis();
  renderDagView();
  renderTree();
  renderMermaidView();
}

function handleFileInput(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const parsed = JSON.parse(e.target.result);
      loadGraphData(parsed);
    } catch (err) {
      alert('Failed to parse JSON file: ' + err.message);
    }
  };
  reader.readAsText(file);
}

// ---------------------------------------------------------------------------
// 2. Tab Navigation & Utilities
// ---------------------------------------------------------------------------

function switchMainTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));
  const btn = document.getElementById('btn-tab-' + tab);
  const view = document.getElementById('view-' + tab);
  if (btn) btn.classList.add('active');
  if (view) view.classList.add('active');
  if (tab === 'mermaid') renderMermaidView();
  if (tab === 'dag') renderDagView();
  if (tab === 'tree') renderTree();
}

function copyMermaidCode() {
  const code = document.getElementById('mermaid-code').textContent;
  navigator.clipboard.writeText('```mermaid\n' + code + '```').then(() => {
    alert('✅ Mermaid Markdown copied to clipboard!');
  }).catch(() => {
    alert('Failed to copy to clipboard.');
  });
}

function formatNumber(num) {
  if (num === undefined || num === null) return '0';
  return num.toLocaleString();
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function getNodeOpType(name) {
  const n = name.toLowerCase();
  if (n.endsWith('.core') || n.includes('dot_product') || n.includes('attention_core')) return { icon: '🎯', type: 'Scaled Dot-Product Attention Core', cls: 'attn' };
  if (n.includes('ln_') || n.includes('norm')) return { icon: '📐', type: 'RMSNorm / LayerNorm', cls: 'norm' };
  if (n.includes('wte') || n.includes('wpe') || n.includes('emb')) return { icon: '🔲', type: 'Embedding Table', cls: 'emb' };
  if (n.includes('attn') && !n.includes('q_') && !n.includes('k_') && !n.includes('v_')) return { icon: '🔀', type: 'Causal Self-Attention', cls: 'attn' };
  if (n.includes('mlp')) return { icon: '⚡', type: 'MLP / SwiGLU Block', cls: 'mlp' };
  if (n.includes('layers') && !n.match(/layers\.\d+/)) return { icon: '🥞', type: 'Transformer Decoder Stack', cls: 'attn' };
  if (n.includes('q_attn') || n.includes('k_attn') || n.includes('v_attn')) return { icon: '⚙️', type: 'Multi-Head Linear Proj', cls: 'linear' };
  if (n.includes('c_proj') || n.includes('c_fc') || n.includes('lm_head') || n.includes('linear')) return { icon: '⚙️', type: 'Linear (Dense)', cls: 'linear' };
  if (n.includes('add') || n.includes('residual')) return { icon: '⊕', type: 'Residual Add', cls: 'add' };
  return { icon: '📦', type: 'Module Block', cls: 'linear' };
}

// ---------------------------------------------------------------------------
// 3. Formula Resolution
// ---------------------------------------------------------------------------

function findModuleByPath(root, path) {
  if (!root) return null;
  if (!path || path === 'root' || path === '(Root / Global Scope)') {
    if (root.path === '' || root.name === 'root') return root;
  }
  if (root.path === path) return root;
  if (root.children) {
    for (const child of root.children) {
      const found = findModuleByPath(child, path);
      if (found) return found;
    }
  }
  return null;
}

function getEffectiveFormula(key) {
  // 0. Direct match from Module or Op object provided directly by backend JSON
  const mod = findModuleByPath(RAW_GRAPH ? RAW_GRAPH.root : null, key);
  if (mod && mod.formula) return { formula: mod.formula, source: 'CODE_SPECIFIED' };
  const matchedOp = OPS_DATA.find(o => o.name === key);
  if (matchedOp && matchedOp.formula) return { formula: matchedOp.formula, source: 'OP_TYPE' };

  // 1. Exact match from JSON metadata (generated by Zig backend)
  if (FORMULAS_DATA[key]) return { formula: FORMULAS_DATA[key], source: 'CODE_SPECIFIED' };

  // 2. Exact match from Ops data: check if this key represents a specific operation output node
  if (matchedOp && FORMULAS_DATA[matchedOp.op_type]) {
    return { formula: FORMULAS_DATA[matchedOp.op_type], source: 'OP_TYPE' };
  }

  // 3. Sub-path prefix inheritance: find LONGEST matching parent prefix
  let bestParent = null;
  let bestParentLen = 0;
  for (const parentKey in FORMULAS_DATA) {
    if (key.startsWith(parentKey + '.')) {
      if (parentKey.length > bestParentLen) {
        bestParentLen = parentKey.length;
        bestParent = parentKey;
      }
    }
  }
  if (bestParent) {
    return { formula: FORMULAS_DATA[bestParent], source: 'INHERITED' };
  }

  // 4. Match from Ops data by module name
  const op = OPS_DATA.find(o => o.module === key);
  if (op && (op.formula || FORMULAS_DATA[op.op_type])) {
    return { formula: op.formula || FORMULAS_DATA[op.op_type], source: 'OP_TYPE' };
  }

  return null;
}

// ---------------------------------------------------------------------------
// 4. Modal Node Inspector
// ---------------------------------------------------------------------------

let CURRENT_INSP_KEY = null;
let CURRENT_INSP_TAB = 'overview';
const INSP_HISTORY_STACK = [];

function switchInspectorTab(tab) {
  CURRENT_INSP_TAB = tab;
  ['overview', 'submodules', 'params'].forEach(t => {
    const btn = document.getElementById('btn-insp-' + t);
    const view = document.getElementById('insp-view-' + t);
    if (btn) btn.classList.toggle('active', t === tab);
    if (view) view.classList.toggle('active', t === tab);
  });
}

function inspectorGoBack() {
  if (INSP_HISTORY_STACK.length > 1) {
    // Pop current
    INSP_HISTORY_STACK.pop();
    // Pop parent to re-open
    const prev = INSP_HISTORY_STACK.pop();
    openInspector(prev.key, prev.tab);
  }
}

function openInspector(rawKey, initialTab = 'overview') {
  // Normalize root key variants and strip any redundant 'root.' prefix
  let key = rawKey;
  if (!key || key === 'root' || key === '(Root / Global Scope)' || key === '') {
    key = 'root';
  } else if (key.startsWith('root.')) {
    key = key.slice(5) || 'root';
  }

  // Maintain history stack for backwards navigation
  if (INSP_HISTORY_STACK.length === 0 || INSP_HISTORY_STACK[INSP_HISTORY_STACK.length - 1].key !== key) {
    INSP_HISTORY_STACK.push({ key, tab: initialTab });
  }

  CURRENT_INSP_KEY = key;
  const modal = document.getElementById('inspector-overlay');
  const titleEl = document.getElementById('insp-title');
  const subEl = document.getElementById('insp-subtitle');
  const iconEl = document.getElementById('insp-icon');
  const bodyEl = document.getElementById('insp-body');
  const breadcrumbEl = document.getElementById('insp-breadcrumb-bar');
  if (!modal || !bodyEl) return;

  const isAttentionCore = key.endsWith('.core');
  const baseKey = isAttentionCore ? key.slice(0, -5) : key;
  const isRoot = (key === 'root');

  // Render Hierarchical Breadcrumb Navigation Bar (Clean root handling without root/root duplicates)
  if (breadcrumbEl) {
    let bcHtml = '';
    if (INSP_HISTORY_STACK.length > 1) {
      bcHtml += '<button class="insp-back-btn" onclick="inspectorGoBack()">← Back</button>';
    }

    if (isRoot) {
      bcHtml += '<span class="insp-breadcrumb-item current">root (Model Overview)</span>';
    } else {
      bcHtml += '<span class="insp-breadcrumb-item" onclick="openInspector(\'root\', \'submodules\')">root</span>';
      const parts = key.split('.');
      let accum = '';
      parts.forEach((p, idx) => {
        accum = accum ? `${accum}.${p}` : p;
        const isLast = (idx === parts.length - 1);
        bcHtml += '<span class="insp-breadcrumb-sep">/</span>';
        if (isLast) {
          bcHtml += `<span class="insp-breadcrumb-item current">${p}</span>`;
        } else {
          const segPath = accum;
          bcHtml += `<span class="insp-breadcrumb-item" onclick="openInspector('${segPath}', 'submodules')">${p}</span>`;
        }
      });
    }
    breadcrumbEl.innerHTML = bcHtml;
  }

  const targetMod = findModuleByPath(RAW_GRAPH ? RAW_GRAPH.root : null, key);

  let matchedNodes;
  let matchedOps;
  if (isAttentionCore) {
    matchedNodes = NODES_DATA.filter(n => {
      if (!n.name.startsWith(baseKey + '.')) return false;
      const sub = n.name.slice(baseKey.length + 1);
      return !sub.startsWith('q_attn.') && !sub.startsWith('k_attn.') && !sub.startsWith('v_attn.') && !sub.startsWith('c_proj.');
    });
    matchedOps = OPS_DATA.filter(o => o.module === baseKey);
  } else if (targetMod) {
    // If exact Module node exists in the recursive tree, collect all its nodes & ops
    const collected = collectAllFromTree(targetMod);
    matchedNodes = collected.nodes;
    matchedOps = collected.ops;
  } else {
    matchedNodes = NODES_DATA.filter(n => n.name === key || n.name.startsWith(key + '.'));
    matchedOps = OPS_DATA.filter(o => o.module === key || o.module.startsWith(key + '.') || o.name.startsWith(key + '.'));
  }

  const opInfo = getNodeOpType(key);
  if (iconEl) iconEl.textContent = isRoot ? '🌐' : opInfo.icon;
  if (titleEl) {
    if (isRoot) {
      titleEl.textContent = (RAW_GRAPH && RAW_GRAPH.model_name) ? `${RAW_GRAPH.model_name} (Root Model)` : 'Model Root Overview';
    } else {
      titleEl.textContent = isAttentionCore ? `${baseKey} · Attention Core` : key;
    }
  }

  let totalParams = 0;
  let totalBytes = 0;
  if (targetMod && (targetMod.total_params !== undefined || targetMod.param_count !== undefined)) {
    totalParams = targetMod.total_params || 0;
    totalBytes = targetMod.total_bytes || 0;
  } else {
    matchedNodes.forEach(n => {
      if (n.kind === 'Param') totalParams += n.elements;
      totalBytes += n.bytes;
    });
  }

  const paramStr = totalParams > 0 ? `${formatNumber(totalParams)} params (${formatBytes(totalBytes)})` : '0 params (Parameter-free)';
  if (subEl) subEl.textContent = `${targetMod && targetMod.module_type ? targetMod.module_type : opInfo.type} · ${paramStr}`;

  // Update tab param badge
  const tabParamCount = document.getElementById('insp-tab-param-count');
  if (tabParamCount) tabParamCount.textContent = formatNumber(totalParams);

  // Input & Output Shape Resolution from real graph data
  const isRootModule = isRoot || (!isAttentionCore && (key === 'gpt' || key === 'model' || !key.includes('.')));
  const incEdges = (isAttentionCore || isRoot) ? [] : EDGES_DATA.filter(e => e.to === key || e.to.startsWith(key + '.'));
  const outEdges = (isAttentionCore || isRoot) ? [] : EDGES_DATA.filter(e => e.from === key || e.from.startsWith(key + '.'));

  let modInpShape = 'Unknown';
  let modOutShape = 'Unknown';

  if (isAttentionCore) {
    const firstOp = matchedOps.length > 0 ? matchedOps[0] : null;
    const lastOp = matchedOps.length > 0 ? matchedOps[matchedOps.length - 1] : null;
    const shapeMatch = firstOp && firstOp.input_shape ? firstOp.input_shape.match(/\[[^\]]+\]/) : null;
    modInpShape = shapeMatch ? `Q, K, V: ${shapeMatch[0]} & Mask` : 'Q, K, V Heads & Mask';
    modOutShape = lastOp ? `${lastOp.output_shape} (Concatenated Context)` : '[B, T, D]';
  } else if (isRootModule) {
    const inputNode = NODES_DATA.find(n => n.kind === 'Input');
    if (inputNode) modInpShape = `${inputNode.shape} (Input Tensor)`;
    const outputNode = NODES_DATA.find(n => n.name.includes('logits') || n.name.includes('output'));
    if (outputNode) modOutShape = `${outputNode.shape} (Model Output)`;
  }

  if (modInpShape === 'Unknown') {
    if (incEdges.length > 0) modInpShape = incEdges[0].shape;
    else if (matchedOps.length > 0) modInpShape = matchedOps[0].input_shape;
    else if (matchedNodes.length > 0) modInpShape = matchedNodes[0].shape;
  }
  if (modOutShape === 'Unknown') {
    if (outEdges.length > 0) modOutShape = outEdges[outEdges.length - 1].shape;
    else if (matchedOps.length > 0) modOutShape = matchedOps[matchedOps.length - 1].output_shape;
    else if (matchedNodes.length > 0) modOutShape = matchedNodes[matchedNodes.length - 1].shape;
  }

  let ioBannerHtml = `
    <div class="insp-io-banner">
      <div class="insp-io-card input">
        <div class="insp-io-label">📥 Module Input Shape</div>
        <div class="insp-io-shape">${modInpShape}</div>
        <div class="insp-io-sub">Incoming tensor fed into ${isAttentionCore ? baseKey + ' core' : key}</div>
      </div>
      <div class="insp-io-arrow">➔</div>
      <div class="insp-io-card output">
        <div class="insp-io-label">📤 Module Output Shape</div>
        <div class="insp-io-shape">${modOutShape}</div>
        <div class="insp-io-sub">Final tensor output from ${isAttentionCore ? baseKey + ' core' : key}</div>
      </div>
    </div>
  `;

  // Mathematical Formula Display (Sourced directly from backend JSON)
  const formObj = getEffectiveFormula(key);
  let formulaHtml = '';
  if (formObj && formObj.formula) {
    const isSpecified = formObj.source === 'CODE_SPECIFIED';
    const badgeLabel = isSpecified ? 'SPECIFIED' : 'GRAPH INFERRED';
    const badgeCls = isSpecified ? 'insp-formula-badge specified' : 'insp-formula-badge inferred';
    formulaHtml = `
      <div class="insp-formula-box">
        <div class="insp-formula-header">
          <span>📐 Mathematical Vector Transformation Formula</span>
          <span class="${badgeCls}">${badgeLabel}</span>
        </div>
        <div class="insp-formula-display">
          <span style="font-size:18px;">📐</span>
          <span id="insp-katex-target" style="flex:1;">${formObj.formula}</span>
        </div>
        <div class="insp-formula-desc">Mathematical relationship mapping input representations to output activations.</div>
      </div>
    `;
  }

  // Operations Table
  let opsHtml = '';
  if (matchedOps.length > 0) {
    opsHtml = `
      <div class="insp-section">
        <div class="insp-section-title">
          <span>⚡ Operations & Layer Dimensions</span>
          <span style="font-size: 11px; font-weight: normal; color: #94a3b8;">${matchedOps.length} Operations executed</span>
        </div>
        <table class="node-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Operation / Node Name</th>
              <th>Op Type</th>
              <th style="color:#38bdf8;">📥 Input Shape</th>
              <th style="color:#fbbf24;">⚙️ Param Shape</th>
              <th style="color:#34d399;">📤 Output Shape</th>
              <th>Memory</th>
            </tr>
          </thead>
          <tbody>
    `;
    matchedOps.forEach((op, idx) => {
      opsHtml += `
        <tr>
          <td style="font-family:var(--font-mono); color: #64748b;">${idx + 1}</td>
          <td class="node-name">${op.name}</td>
          <td><span class="badge badge-op">${op.op_type}</span></td>
          <td class="node-shape" style="color:#38bdf8;">${op.input_shape || '-'}</td>
          <td class="node-shape" style="color:#fbbf24;">${op.param_shape || '-'}</td>
          <td class="node-shape" style="color:#34d399;">${op.output_shape || '-'}</td>
          <td style="font-family:var(--font-mono); font-size:12px;">${formatBytes(op.bytes)}</td>
        </tr>
      `;
    });
    opsHtml += '</tbody></table></div>';
  }

  // Inbound & Downstream Connections
  const incoming = isAttentionCore ? [] : EDGES_DATA.filter(e => e.to === key || e.to.startsWith(key + '.') || (key.startsWith(e.to) && e.to.length > 5));
  let inputsHtml = '<div class="insp-section"><div class="insp-section-title"><span>📥 Inbound Connections</span></div>';
  if (incoming.length > 0) {
    inputsHtml += '<table class="node-table"><thead><tr><th>Source Tensor / Predecessor</th><th>Target Port</th><th>Shape</th><th>Connection Type</th></tr></thead><tbody>';
    incoming.forEach(e => {
      const typeBadge = e.is_skip 
        ? '<span class="badge" style="background:#0284c7;color:#fff;">⚡ Residual Skip Highway</span>'
        : '<span class="badge" style="background:#334155;color:#94a3b8;">▼ Sequential Flow</span>';
      inputsHtml += `<tr><td class="node-name">${e.from}</td><td class="node-name">${e.to}</td><td class="node-shape">${e.shape}</td><td>${typeBadge}</td></tr>`;
    });
    inputsHtml += '</tbody></table>';
  } else {
    inputsHtml += '<div style="font-size: 12px; color: #94a3b8; font-style: italic;">Root level input or graph source.</div>';
  }
  inputsHtml += '</div>';

  let outHtml = '';
  if (!isAttentionCore) {
    const outgoing = EDGES_DATA.filter(e => e.from === key || e.from.startsWith(key + '.'));
    if (outgoing.length > 0) {
      outHtml += '<div class="insp-section"><div class="insp-section-title"><span>🔀 Downstream Flow</span></div><table class="node-table"><thead><tr><th>Consumer Node</th><th>Shape</th><th>Connection Type</th></tr></thead><tbody>';
      outgoing.forEach(e => {
        const typeBadge = e.is_skip 
          ? '<span class="badge" style="background:#0284c7;color:#fff;">⚡ Residual Skip Highway</span>'
          : '<span class="badge" style="background:#334155;color:#94a3b8;">▼ Sequential Flow</span>';
        outHtml += `<tr><td class="node-name">${e.to}</td><td class="node-shape">${e.shape}</td><td>${typeBadge}</td></tr>`;
      });
      outHtml += '</tbody></table></div>';
    }
  }

  // --- Submodules Structure Drill-down ---
  const subModulesMap = new Map();
  if (targetMod && targetMod.children && targetMod.children.length > 0) {
    // 1. Direct children from the recursive tree
    targetMod.children.forEach(c => {
      subModulesMap.set(c.path, {
        name: c.name,
        fullPath: c.path,
        params: c.total_params || 0,
        bytes: c.total_bytes || 0,
        count: c.node_count || 0
      });
    });
  } else {
    // 2. Fallback to path prefix scanning on NODES_DATA
    NODES_DATA.forEach(n => {
      if (n.name.startsWith(key + '.')) {
        const remainder = n.name.slice(key.length + 1);
        const childPart = remainder.split('.')[0];
        const childFullPath = `${key}.${childPart}`;
        if (!subModulesMap.has(childFullPath)) {
          subModulesMap.set(childFullPath, { name: childPart, fullPath: childFullPath, params: 0, bytes: 0, count: 0 });
        }
        const item = subModulesMap.get(childFullPath);
        item.count++;
        item.bytes += n.bytes;
        if (n.kind === 'Param') item.params += n.elements;
      }
    });
  }

  let submodulesHtml = '<div class="insp-section"><div class="insp-section-title"><span>📁 Direct Submodules & Internal Hierarchy</span><span style="font-size: 11px; font-weight: normal; color: #94a3b8;">' + subModulesMap.size + ' Submodules</span></div>';
  if (subModulesMap.size > 0) {
    submodulesHtml += renderSubmoduleFlowHtml(key, Array.from(subModulesMap.values()), matchedNodes, matchedOps);
  } else {
    submodulesHtml += '<div style="font-size: 12px; color: #94a3b8; font-style: italic;">This node is a leaf operation or has no children.</div>';
  }
  submodulesHtml += '</div>';


  // --- Parameters Dedicated Inspector Tab ---
  const paramNodes = matchedNodes.filter(n => n.kind === 'Param');
  let paramsHtml = `<div class="insp-section"><div class="insp-section-title"><span>⚙️ Parameter Matrix & Initialization Details</span><span style="font-size: 11px; font-weight: normal; color: #94a3b8;">Total Elements: ${formatNumber(totalParams)} (${formatBytes(totalBytes)})</span></div>`;
  if (paramNodes.length > 0) {
    paramsHtml += '<table class="node-table"><thead><tr><th>Parameter Name</th><th>Shape</th><th>Elements</th><th>Memory</th><th>Init Strategy</th><th>Status</th></tr></thead><tbody>';
    paramNodes.forEach(p => {
      const statusBadge = p.status === 'CUSTOM_INIT' ? 'badge-custom' : 'badge-auto';
      paramsHtml += `<tr><td class="node-name">${p.name}</td><td class="node-shape">${p.shape}</td><td style="font-family:var(--font-mono);">${formatNumber(p.elements)}</td><td style="font-family:var(--font-mono);">${formatBytes(p.bytes)}</td><td class="strategy-col">${p.strategy}</td><td><span class="badge ${statusBadge}">${p.status}</span></td></tr>`;
    });
    paramsHtml += '</tbody></table>';
  } else {
    paramsHtml += '<div style="font-size: 12px; color: #94a3b8; font-style: italic;">No trainable parameters (stateless / parameter-free operation).</div>';
  }
  paramsHtml += '</div>';

  // Compose 3 Tabs
  bodyEl.innerHTML = `
    <div id="insp-view-overview" class="insp-tab-view ${initialTab === 'overview' ? 'active' : ''}">
      ${ioBannerHtml}
      ${formulaHtml}
      ${opsHtml}
      ${inputsHtml}
      ${outHtml}
    </div>
    <div id="insp-view-submodules" class="insp-tab-view ${initialTab === 'submodules' ? 'active' : ''}">
      ${submodulesHtml}
      ${opsHtml}
    </div>
    <div id="insp-view-params" class="insp-tab-view ${initialTab === 'params' ? 'active' : ''}">
      ${paramsHtml}
    </div>
  `;

  modal.classList.add('open');
  switchInspectorTab(initialTab);

  // KaTeX Formula Render
  try {
    const target = document.getElementById('insp-katex-target');
    if (target && window.katex && formObj && formObj.formula) {
      katex.render(formObj.formula, target, {
        throwOnError: false,
        displayMode: true
      });
    }
  } catch(err) {
    console.warn('KaTeX render fallback:', err);
  }
}

function closeInspector(e) {
  if (e && e.target && e.target.id !== 'inspector-overlay' && !e.target.classList.contains('inspector-close-btn')) return;
  const modal = document.getElementById('inspector-overlay');
  if (modal) modal.classList.remove('open');
  INSP_HISTORY_STACK.length = 0;
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('inspector-overlay');
    if (modal) modal.classList.remove('open');
    INSP_HISTORY_STACK.length = 0;
  }
});

// ---------------------------------------------------------------------------
// 4.1 Internal Submodules DAG Flowchart Renderer (Matches Main Architecture DAG)
// ---------------------------------------------------------------------------

function renderSubmoduleFlowHtml(parentKey, submodulesList, matchedNodes, matchedOps) {
  if (!submodulesList || submodulesList.length === 0) {
    return '<div style="font-size: 12px; color: #94a3b8; font-style: italic;">No submodules present.</div>';
  }

  // Map of submodule fullPath / name -> sub object
  const subMap = new Map();
  submodulesList.forEach(s => {
    subMap.set(s.fullPath, s);
    subMap.set(s.name, s);
  });

  // Determine connections among direct submodules
  // Check EDGES_DATA for direct connections or connections via internal operations
  const subEdges = [];
  const edgeKeysSeen = new Set();

  const currentMod = findModuleByPath(RAW_GRAPH ? RAW_GRAPH.root : null, parentKey);
  const edgesToScan = (currentMod && currentMod.edges && currentMod.edges.length > 0)
    ? currentMod.edges.concat(EDGES_DATA)
    : EDGES_DATA;

  edgesToScan.forEach(e => {
    let fromSub = null;
    let toSub = null;

    if (subMap.has(e.from)) {
      fromSub = subMap.get(e.from).fullPath;
    } else if (parentKey === 'root') {
      const topPart = e.from.split('.')[0];
      if (subMap.has(topPart)) fromSub = subMap.get(topPart).fullPath;
    } else if (e.from.startsWith(parentKey + '.')) {
      const rest = e.from.slice(parentKey.length + 1).split('.')[0];
      const candidate = `${parentKey}.${rest}`;
      if (subMap.has(candidate)) fromSub = candidate;
    }

    if (subMap.has(e.to)) {
      toSub = subMap.get(e.to).fullPath;
    } else if (parentKey === 'root') {
      const topPart = e.to.split('.')[0];
      if (subMap.has(topPart)) toSub = subMap.get(topPart).fullPath;
    } else if (e.to.startsWith(parentKey + '.')) {
      const rest = e.to.slice(parentKey.length + 1).split('.')[0];
      const candidate = `${parentKey}.${rest}`;
      if (subMap.has(candidate)) toSub = candidate;
    }

    if (fromSub && toSub && fromSub !== toSub) {
      const ek = `${fromSub}->${toSub}`;
      if (!edgeKeysSeen.has(ek)) {
        edgeKeysSeen.add(ek);
        subEdges.push({
          from: fromSub,
          to: toSub,
          shape: e.shape,
          is_skip: e.is_skip
        });
      }
    }
  });

  // Also check if any submodule (e.g. residual_attn, output) acts as a converge node with skip connections
  const incoming = {};
  const outgoing = {};
  const inDegree = new Map();
  submodulesList.forEach(s => {
    incoming[s.fullPath] = [];
    outgoing[s.fullPath] = [];
    inDegree.set(s.fullPath, 0);
  });

  subEdges.forEach(e => {
    if (incoming[e.to]) incoming[e.to].push(e);
    if (outgoing[e.from]) outgoing[e.from].push(e);
    inDegree.set(e.to, (inDegree.get(e.to) || 0) + 1);
  });

  // Special case: Attention submodule container with q, k, v, c_proj
  const isAttnContainer = submodulesList.some(s => s.name.startsWith('q_') || s.name.startsWith('k_') || s.name.startsWith('v_'));
  if (isAttnContainer) {
    const q = submodulesList.find(s => s.name.startsWith('q_'));
    const k = submodulesList.find(s => s.name.startsWith('k_'));
    const v = submodulesList.find(s => s.name.startsWith('v_'));
    const proj = submodulesList.find(s => s.name.includes('proj'));
    const others = submodulesList.filter(s => s !== q && s !== k && s !== v && s !== proj);

    // Render parallel Q/K/V branches followed by Attention Core / Proj
    let html = '<div class="submod-flow-container">';
    html += `
      <div class="flow-stage-parallel">
        <div class="parallel-header">
          <span>⚡ Multi-Head Projections (Parallel Q, K, V)</span>
          <span class="flow-card-shape">${q && q.params > 0 ? `${formatNumber(q.params)} params each` : 'Parallel'}</span>
        </div>
        <div class="parallel-branches-row">
    `;
    [q, k, v].filter(Boolean).forEach((bSub, bIdx) => {
      if (bIdx > 0) html += '<div class="branch-divider">|</div>';
      const bOp = getNodeOpType(bSub.fullPath);
      html += `
        <div class="parallel-branch-col transform-branch" onclick="openInspector('${bSub.fullPath}')">
          <span class="branch-badge transform">${bSub.name.toUpperCase()} Branch</span>
          <div style="margin-top: 8px;">
            <div class="flow-card-title">${bSub.name}</div>
            <div style="font-size: 11px; color: #a5b4fc; margin-top: 4px;">${bOp.type}</div>
            <div style="font-size: 11px; color: var(--text-sub); margin-top: 2px;">
              ${bSub.params > 0 ? `⚙️ ${formatNumber(bSub.params)} params` : '⚡ Stateless'}
            </div>
            <div class="dag-card-actions">
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${bSub.fullPath}', 'overview')">📐 Formula</button>
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${bSub.fullPath}', 'submodules')">📁 Submodules</button>
              ${bSub.params > 0 ? `<button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${bSub.fullPath}', 'params')">⚙️ Params</button>` : ''}
            </div>
          </div>
        </div>
      `;
    });
    html += `
        </div>
        <div class="converge-arrow-box">
          <div class="flow-arrow-down">
            <span class="flow-arrow-head">▼</span>
            <span class="flow-arrow-label">🎯 Scaled Dot-Product Core: Softmax(Q·Kᵀ / √d)·V</span>
          </div>
        </div>
      </div>
    `;

    if (proj) {
      html += `
        <div class="flow-arrow-down">
          <div class="flow-arrow-line"></div>
          <span class="flow-arrow-head">▼</span>
        </div>
      `;
      const projOp = getNodeOpType(proj.fullPath);
      html += `
        <div class="flow-card flow-card-linear" onclick="openInspector('${proj.fullPath}')">
          <div class="flow-card-header">
            <span class="flow-card-title">${proj.name}</span>
            <span class="flow-card-shape">${proj.params > 0 ? `${formatNumber(proj.params)} params` : 'Linear'}</span>
          </div>
          <div class="flow-card-body">
            <span>${projOp.type} · Output Projection</span>
            <div class="dag-card-actions" style="margin-top: 0;">
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${proj.fullPath}', 'overview')">📐 Formula</button>
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${proj.fullPath}', 'submodules')">📁 Submodules</button>
              ${proj.params > 0 ? `<button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${proj.fullPath}', 'params')">⚙️ Params</button>` : ''}
            </div>
          </div>
        </div>
      `;
    }

    others.forEach(oth => {
      html += `
        <div class="flow-arrow-down">
          <div class="flow-arrow-line"></div>
          <span class="flow-arrow-head">▼</span>
        </div>
      `;
      const othOp = getNodeOpType(oth.fullPath);
      html += `
        <div class="flow-card flow-card-linear" onclick="openInspector('${oth.fullPath}')">
          <div class="flow-card-header">
            <span class="flow-card-title">${oth.name}</span>
            <span class="flow-card-shape">${oth.params > 0 ? `${formatNumber(oth.params)} params` : 'Module'}</span>
          </div>
          <div class="flow-card-body">
            <span>${othOp.type}</span>
            <div class="dag-card-actions" style="margin-top: 0;">
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${oth.fullPath}', 'overview')">📐 Formula</button>
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${oth.fullPath}', 'submodules')">📁 Submodules</button>
              ${oth.params > 0 ? `<button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${oth.fullPath}', 'params')">⚙️ Params</button>` : ''}
            </div>
          </div>
        </div>
      `;
    });

    html += '</div>';
    return html;
  }

  // Fallback to topological ordering of submodules if edges exist, or default sequential
  let orderedSubmodules = [];
  if (subEdges.length > 0) {
    const queue = [];
    inDegree.forEach((deg, node) => {
      if (deg === 0) queue.push(node);
    });

    const tempInDegree = new Map(inDegree);
    while (queue.length > 0) {
      const u = queue.shift();
      if (subMap.has(u)) orderedSubmodules.push(subMap.get(u));
      const neighbors = outgoing[u] || [];
      neighbors.forEach(e => {
        const v = e.to;
        tempInDegree.set(v, tempInDegree.get(v) - 1);
        if (tempInDegree.get(v) === 0) {
          queue.push(v);
        }
      });
    }

    submodulesList.forEach(s => {
      if (!orderedSubmodules.some(o => o.fullPath === s.fullPath)) {
        orderedSubmodules.push(s);
      }
    });
  } else {
    orderedSubmodules = submodulesList.slice();
  }


  // Render orderedSubmodules into DAG Flow Pipeline
  let html = '<div class="submod-flow-container">';
  orderedSubmodules.forEach((sub, sIdx) => {
    const inc = incoming[sub.fullPath] || [];
    const hasSkip = inc.some(e => e.is_skip);
    const isMultiBranch = inc.length > 1;
    const subOp = getNodeOpType(sub.fullPath);

    if (isMultiBranch && hasSkip) {
      const skipEdge = inc.find(e => e.is_skip);
      const transformEdge = inc.find(e => !e.is_skip);
      const skipFrom = skipEdge ? skipEdge.from : sub.fullPath;
      const transFrom = transformEdge ? transformEdge.from : sub.fullPath;
      const skipSubName = skipFrom.split('.').pop();
      const transSubName = transFrom.split('.').pop();

      html += `
        <div class="flow-stage-parallel">
          <div class="parallel-header">
            <span>Residual Connection · ${sub.name}</span>
            <span class="flow-card-shape">${skipEdge ? skipEdge.shape : ''}</span>
          </div>
          <div class="parallel-branches-row">
            <div class="parallel-branch-col skip-branch" onclick="openInspector('${skipFrom}')">
              <span class="branch-badge skip">⚡ Shortcut (Skip)</span>
              <div style="margin-top: 8px;">
                <div class="flow-card-title">${skipSubName}</div>
                <div style="font-size: 11px; color: #38bdf8; margin-top: 4px;">Direct Identity Highway</div>
                <div style="font-size: 11px; color: var(--text-sub); margin-top: 2px;">Preserves gradient flow</div>
                <div class="dag-card-actions">
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${skipFrom}', 'overview')">📐 Formula</button>
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${skipFrom}', 'submodules')">📁 Submodules</button>
                </div>
              </div>
            </div>

            <div class="branch-divider">|</div>

            <div class="parallel-branch-col transform-branch" onclick="openInspector('${transFrom}')">
              <span class="branch-badge transform">⚙️ Transform Branch</span>
              <div style="margin-top: 8px;">
                <div class="flow-card-title">${transSubName}</div>
                <div style="font-size: 11px; color: #a5b4fc; margin-top: 4px;">Layer Transform</div>
                <div style="font-size: 11px; color: var(--text-sub); margin-top: 2px;">F(x) Feature Extraction</div>
                <div class="dag-card-actions">
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${transFrom}', 'overview')">📐 Formula</button>
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${transFrom}', 'submodules')">📁 Submodules</button>
                  <button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${transFrom}', 'params')">⚙️ Params</button>
                </div>
              </div>
            </div>
          </div>

          <div class="converge-arrow-box">
            <div class="flow-arrow-down">
              <span class="flow-arrow-head">▼</span>
              <span class="flow-arrow-label">⊕ Element-wise Add (x + F(x))</span>
            </div>
            <div class="converge-card" onclick="openInspector('${sub.fullPath}')">
              <div class="converge-title">
                <span>⊕</span>
                <span>${sub.name}</span>
              </div>
              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">
                Residual Sum Output · Shape: <span class="flow-card-shape">${skipEdge ? skipEdge.shape : ''}</span>
              </div>
              <div class="dag-card-actions" style="justify-content: center; margin-top: 6px;">
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${sub.fullPath}', 'overview')">📐 View Formula</button>
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${sub.fullPath}', 'submodules')">📁 View Submodules</button>
              </div>
            </div>
          </div>
        </div>
      `;
    } else if (isMultiBranch && !hasSkip) {
      html += `
        <div class="flow-stage-parallel">
          <div class="parallel-header">
            <span>Parallel Inputs · ${sub.name} Convergence</span>
            <span class="flow-card-shape">${inc[0] ? inc[0].shape : ''}</span>
          </div>
          <div class="parallel-branches-row">
      `;
      inc.forEach((e, bIdx) => {
        if (bIdx > 0) html += '<div class="branch-divider">|</div>';
        const fromSubName = e.from.split('.').pop();
        html += `
          <div class="parallel-branch-col" onclick="openInspector('${e.from}')">
            <span class="branch-badge transform">Branch ${bIdx + 1}</span>
            <div style="margin-top: 8px;">
              <div class="flow-card-title">${fromSubName}</div>
              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">Shape: ${e.shape || '-'}</div>
              <div class="dag-card-actions">
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${e.from}', 'overview')">📐 Formula</button>
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${e.from}', 'submodules')">📁 Submodules</button>
                <button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${e.from}', 'params')">⚙️ Params</button>
              </div>
            </div>
          </div>
        `;
      });
      html += `
          </div>
          <div class="converge-arrow-box">
            <div class="flow-arrow-down">
              <span class="flow-arrow-head">▼</span>
              <span class="flow-arrow-label">⊕ Converge / Sum</span>
            </div>
            <div class="converge-card" onclick="openInspector('${sub.fullPath}')">
              <div class="converge-title">
                <span>⊕</span>
                <span>${sub.name}</span>
              </div>
              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">Combined Output</div>
              <div class="dag-card-actions" style="justify-content: center; margin-top: 6px;">
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${sub.fullPath}', 'overview')">📐 View Formula</button>
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${sub.fullPath}', 'submodules')">📁 View Submodules</button>
              </div>
            </div>
          </div>
        </div>
      `;
    } else {
      // Sequential Stage Card
      const pCount = sub.params || 0;
      const pMeta = pCount > 0 ? `${formatNumber(pCount)} params (${formatBytes(sub.bytes || 0)})` : '⚡ Stateless';
      const edge = inc[0];
      const shapeText = edge ? edge.shape : '';

      html += `
        <div class="flow-card flow-card-linear" onclick="openInspector('${sub.fullPath}')">
          <div class="flow-card-header">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="tb-op-icon ${subOp.cls}">${subOp.icon}</span>
              <span class="flow-card-title">${sub.name}</span>
            </div>
            ${shapeText ? `<span class="flow-card-shape">${shapeText}</span>` : `<span class="badge badge-op">${subOp.type.split('/')[0].trim()}</span>`}
          </div>
          <div class="flow-card-body">
            <span>${pMeta}</span>
            <div class="dag-card-actions" style="margin-top: 0;">
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${sub.fullPath}', 'overview')">📐 Formula</button>
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${sub.fullPath}', 'submodules')">📁 Submodules</button>
              ${pCount > 0 ? `<button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${sub.fullPath}', 'params')">⚙️ Params</button>` : ''}
            </div>
          </div>
        </div>
      `;
    }

    if (sIdx < orderedSubmodules.length - 1) {
      html += `
        <div class="flow-arrow-down">
          <div class="flow-arrow-line"></div>
          <span class="flow-arrow-head">▼</span>
        </div>
      `;
    }
  });

  html += '</div>';
  return html;
}

// ---------------------------------------------------------------------------
// 5. Dynamic DAG Flow & Skip Connection Architecture View
// ---------------------------------------------------------------------------

function renderDagView() {
  const container = document.getElementById('dag-container');
  if (!container) return;

  const modParams = {};
  NODES_DATA.forEach(n => {
    const mod = n.name.split('.').slice(0, -1).join('.');
    if (!modParams[mod]) modParams[mod] = 0;
    if (n.kind === 'Param') modParams[mod] += n.elements;
  });

  const incoming = {};
  const outgoing = {};
  const inDegree = new Map();
  const allNodes = new Set();

  EDGES_DATA.forEach(e => {
    allNodes.add(e.from);
    allNodes.add(e.to);
    if (!incoming[e.to]) incoming[e.to] = [];
    incoming[e.to].push(e);
    if (!outgoing[e.from]) outgoing[e.from] = [];
    outgoing[e.from].push(e);
    inDegree.set(e.to, (inDegree.get(e.to) || 0) + 1);
    if (!inDegree.has(e.from)) inDegree.set(e.from, 0);
  });

  // Dynamic Topological Ordering (Kahn's algorithm)
  const queue = [];
  inDegree.forEach((deg, node) => {
    if (deg === 0) queue.push(node);
  });

  const topoOrder = [];
  const tempInDegree = new Map(inDegree);
  while (queue.length > 0) {
    const u = queue.shift();
    topoOrder.push(u);
    const neighbors = outgoing[u] || [];
    neighbors.forEach(e => {
      const v = e.to;
      tempInDegree.set(v, tempInDegree.get(v) - 1);
      if (tempInDegree.get(v) === 0) {
        queue.push(v);
      }
    });
  }

  // 1. Build Macro Structural Stages:
  // Identify residual summation/converge nodes as primary milestones, plus root inputs and output heads
  const macroStages = topoOrder.filter(n => {
    // Top-level input summation
    if (n.includes('embeddings_sum') || n === 'inputs.token_ids') return true;
    // Layer residual summation points
    if (n.endsWith('.residual_attn') || n.endsWith('.output')) return true;
    // Final norm and head
    if (n.endsWith('.ln_f') || n.endsWith('.lm_head') || n.endsWith('.logits') || n.startsWith('outputs.')) return true;
    return false;
  });

  const stages = macroStages.length > 0 ? macroStages : topoOrder.filter(n => incoming[n] && incoming[n].length > 0);

  let html = '';
  stages.forEach((target, sIdx) => {
    const inc = incoming[target] || [];
    const hasSkip = inc.some(e => e.is_skip);
    const isMultiBranch = inc.length > 1;

    if (hasSkip) {
      const skipEdge = inc.find(e => e.is_skip);
      const transformEdge = inc.find(e => !e.is_skip);

      const isAttnBlock = target.endsWith('.residual_attn');
      const blockLabel = isAttnBlock ? 'Residual Attention Sub-Layer' : 'Residual Feed-Forward Sub-Layer';
      const layerMatch = target.match(/layers\.(\d+)/);
      const layerNum = layerMatch ? `Layer ${layerMatch[1]}` : 'Block';
      const skipFrom = skipEdge ? skipEdge.from : target;
      
      // Determine what was transformed:
      // If Attention, the transformation pipeline is [ln_1 -> attn]
      // If MLP, the transformation pipeline is [ln_2 -> mlp]
      let transFrom = transformEdge ? transformEdge.from : target;
      let transSubDesc = isAttnBlock ? 'RMSNorm + Multi-Head Attention' : 'RMSNorm + MLP Block';
      if (layerMatch) {
        const prefix = `gpt.layers.${layerMatch[1]}`;
        if (isAttnBlock) {
          transFrom = `${prefix}.attn`;
          transSubDesc = `${prefix}.ln_1 ➔ ${prefix}.attn`;
        } else {
          transFrom = `${prefix}.mlp`;
          transSubDesc = `${prefix}.ln_2 ➔ ${prefix}.mlp`;
        }
      }

      html += `
        <div class="flow-stage-parallel">
          <div class="parallel-header">
            <span>${layerNum} · ${blockLabel}</span>
            <span class="flow-card-shape">${skipEdge ? skipEdge.shape : ''}</span>
          </div>
          <div class="parallel-branches-row">
            <div class="parallel-branch-col skip-branch" onclick="openInspector('${skipFrom}')">
              <span class="branch-badge skip">⚡ Shortcut (Skip)</span>
              <div style="margin-top: 8px;">
                <div class="flow-card-title">${skipFrom}</div>
                <div style="font-size: 11px; color: #38bdf8; margin-top: 4px;">Direct Identity Highway</div>
                <div style="font-size: 11px; color: var(--text-sub); margin-top: 2px;">Preserves gradient highway (x)</div>
                <div class="dag-card-actions">
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${skipFrom}', 'overview')">📐 Formula</button>
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${skipFrom}', 'submodules')">📁 Submodules</button>
                </div>
              </div>
            </div>

            <div class="branch-divider">|</div>

            <div class="parallel-branch-col transform-branch" onclick="openInspector('${transFrom}')">
              <span class="branch-badge transform">⚙️ Transform Branch</span>
              <div style="margin-top: 8px;">
                <div class="flow-card-title">${transFrom}</div>
                <div style="font-size: 11px; color: #a5b4fc; margin-top: 4px;">${transSubDesc}</div>
                <div style="font-size: 11px; color: var(--text-sub); margin-top: 2px;">F(x) Feature Transformation</div>
                <div class="dag-card-actions">
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${transFrom}', 'overview')">📐 Formula</button>
                  <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${transFrom}', 'submodules')">📁 Submodules</button>
                  <button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${transFrom}', 'params')">⚙️ Params</button>
                </div>
              </div>
            </div>
          </div>

          <div class="converge-arrow-box">
            <div class="flow-arrow-down">
              <span class="flow-arrow-head">▼</span>
              <span class="flow-arrow-label">⊕ Element-wise Add (x + F(x))</span>
            </div>
            <div class="converge-card" onclick="openInspector('${target}')">
              <div class="converge-title">
                <span>⊕</span>
                <span>${target}</span>
              </div>
              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">
                Residual Sum Output · Shape: <span class="flow-card-shape">${skipEdge ? skipEdge.shape : ''}</span>
              </div>
              <div class="dag-card-actions" style="justify-content: center; margin-top: 6px;">
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${target}', 'overview')">📐 View Formula</button>
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${target}', 'submodules')">📁 View Submodules</button>
              </div>
            </div>
          </div>
        </div>
      `;
    } else if (isMultiBranch && !hasSkip) {
      // General multi-input parallel convergence (e.g. embeddings sum)
      html += `
        <div class="flow-stage-parallel">
          <div class="parallel-header">
            <span>Parallel Inputs · Stage Convergence</span>
            <span class="flow-card-shape">${inc[0] ? inc[0].shape : ''}</span>
          </div>
          <div class="parallel-branches-row">
      `;
      inc.forEach((e, bIdx) => {
        if (bIdx > 0) html += '<div class="branch-divider">|</div>';
        html += `
          <div class="parallel-branch-col" onclick="openInspector('${e.from}')">
            <span class="branch-badge transform">Branch ${bIdx + 1}</span>
            <div style="margin-top: 8px;">
              <div class="flow-card-title">${e.from}</div>
              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">Shape: ${e.shape || '-'}</div>
              <div class="dag-card-actions">
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${e.from}', 'overview')">📐 Formula</button>
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${e.from}', 'submodules')">📁 Submodules</button>
                <button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${e.from}', 'params')">⚙️ Params</button>
              </div>
            </div>
          </div>
        `;
      });
      html += `
          </div>
          <div class="converge-arrow-box">
            <div class="flow-arrow-down">
              <span class="flow-arrow-head">▼</span>
              <span class="flow-arrow-label">⊕ Converge / Sum</span>
            </div>
            <div class="converge-card" onclick="openInspector('${target}')">
              <div class="converge-title">
                <span>⊕</span>
                <span>${target}</span>
              </div>
              <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px;">Combined Representation</div>
              <div class="dag-card-actions" style="justify-content: center; margin-top: 6px;">
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${target}', 'overview')">📐 View Formula</button>
                <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${target}', 'submodules')">📁 View Submodules</button>
              </div>
            </div>
          </div>
        </div>
      `;
    } else {
      // Sequential Stage
      const pCount = modParams[target] || 0;
      const pMeta = pCount > 0 ? `${formatNumber(pCount)} params` : '';
      const edge = inc[0];
      const shapeText = edge ? edge.shape : '';

      html += `
        <div class="flow-card flow-card-linear" onclick="openInspector('${target}')">
          <div class="flow-card-header">
            <span class="flow-card-title">${target}</span>
            ${shapeText ? `<span class="flow-card-shape">${shapeText}</span>` : ''}
          </div>
          <div class="flow-card-body">
            <span>${pMeta ? `⚙️ ${pMeta}` : 'Forward checkpoint'}</span>
            <div class="dag-card-actions" style="margin-top: 0;">
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${target}', 'overview')">📐 Formula</button>
              <button class="dag-action-btn" onclick="event.stopPropagation(); openInspector('${target}', 'submodules')">📁 Submodules</button>
              ${pCount > 0 ? `<button class="dag-action-btn param-btn" onclick="event.stopPropagation(); openInspector('${target}', 'params')">⚙️ Params</button>` : ''}
            </div>
          </div>
        </div>
      `;
    }

    if (sIdx < stages.length - 1) {
      html += `
        <div class="flow-arrow-down">
          <div class="flow-arrow-line"></div>
          <span class="flow-arrow-head">▼</span>
        </div>
      `;
    }
  });


  container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// 6. Interactive Mermaid Flowchart with Click Callback
// ---------------------------------------------------------------------------

function renderMermaidView() {
  const container = document.getElementById('mermaid-code');
  if (!container) return;

  let code = 'flowchart TD\n';

  const nodeMap = new Map();
  NODES_DATA.forEach(n => nodeMap.set(n.name, n));

  const dagNodes = new Set();
  EDGES_DATA.forEach(e => {
    dagNodes.add(e.from);
    dagNodes.add(e.to);
  });
  if (dagNodes.size === 0) {
    NODES_DATA.forEach(n => dagNodes.add(n.name));
  }

  function formatNodeDef(name) {
    const safeId = 'n_' + name.replace(/[^a-zA-Z0-9_]/g, '_');
    const node = nodeMap.get(name);
    const leafName = (name.split('.').pop() || name).replace(/"/g, "'");
    const shape = node ? (node.shape || '').replace(/"/g, "'") : '';
    const shapeText = shape ? `<br>${shape}` : '';
    const kind = node ? node.kind : '';

    if (kind === 'Input' || name.startsWith('inputs.') || name.includes('input')) {
      return `${safeId}(["📥 ${leafName}${shapeText}"])`;
    } else if (kind === 'Param') {
      return `${safeId}["⚖️ ${leafName}${shapeText}"]`;
    } else if (name.includes('residual') || name.includes('add') || name.includes('skip') || name.includes('sum')) {
      return `${safeId}["⊕ ${leafName}${shapeText}"]`;
    } else {
      return `${safeId}["⚡ ${leafName}${shapeText}"]`;
    }
  }

  function buildScopeTree(names) {
    const root = { _nodes: [], _children: {} };
    names.forEach(name => {
      const parts = name.split('.');
      let curr = root;
      for (let i = 0; i < parts.length - 1; i++) {
        const p = parts[i];
        if (!curr._children[p]) {
          curr._children[p] = { _nodes: [], _children: {} };
        }
        curr = curr._children[p];
      }
      curr._nodes.push(name);
    });
    return root;
  }

  const scopeTree = buildScopeTree(Array.from(dagNodes));

  function renderSubgraphsRecursive(node, currentPath, indent) {
    const ind = '  '.repeat(indent);
    let out = '';
    Object.keys(node._children).sort().forEach(k => {
      const child = node._children[k];
      const childPath = currentPath ? `${currentPath}.${k}` : k;
      const sgId = 'sg_' + childPath.replace(/[^a-zA-Z0-9_]/g, '_');
      const cleanK = k.replace(/"/g, "'");
      out += `${ind}subgraph ${sgId} ["📁 ${cleanK}"]\n`;
      out += renderSubgraphsRecursive(child, childPath, indent + 1);
      out += `${ind}end\n\n`;
    });
    node._nodes.sort().forEach(n => {
      out += `${ind}${formatNodeDef(n)}\n`;
    });
    return out;
  }

  code += renderSubgraphsRecursive(scopeTree, '', 1);

  const edgesSeen = new Set();
  EDGES_DATA.forEach(e => {
    const fromSafe = 'n_' + e.from.replace(/[^a-zA-Z0-9_]/g, '_');
    const toSafe = 'n_' + e.to.replace(/[^a-zA-Z0-9_]/g, '_');
    const edgeKey = fromSafe + '->' + toSafe;
    if (edgesSeen.has(edgeKey)) return;
    edgesSeen.add(edgeKey);

    const label = (e.shape || '').replace(/"/g, "'");
    if (e.is_skip) {
      code += label ? `  ${fromSafe} -.->|"⚡ Skip: ${label}"| ${toSafe}\n` : `  ${fromSafe} -.->|"⚡ Skip"| ${toSafe}\n`;
    } else if (label) {
      code += `  ${fromSafe} -->|"${label}"| ${toSafe}\n`;
    } else {
      code += `  ${fromSafe} --> ${toSafe}\n`;
    }
  });

  // Attach interactive click handlers in Mermaid DSL
  dagNodes.forEach(nodeName => {
    const safeId = 'n_' + nodeName.replace(/[^a-zA-Z0-9_]/g, '_');
    code += `  click ${safeId} call onMermaidNodeClick("${nodeName}")\n`;
  });

  container.textContent = code;

  if (window.mermaid) {
    const renderTarget = document.getElementById('mermaid-diagram-target');
    if (renderTarget) {
      try {
        mermaid.render('mermaid-svg-chart-' + Date.now(), code).then(({ svg }) => {
          renderTarget.innerHTML = svg;
        }).catch(err => {
          console.warn('Mermaid SVG render fallback:', err);
        });
      } catch(err) {
        console.warn('Mermaid execution:', err);
      }
    }
  }
}

// Global click hook for Mermaid SVG Nodes
window.onMermaidNodeClick = function(nodeName) {
  openInspector(nodeName);
};

// ---------------------------------------------------------------------------
// 7. Hierarchical Module Tree View
// ---------------------------------------------------------------------------

function buildHierarchy(data) {
  const root = { _children: {}, _nodes: [], _paramCount: 0, _bytes: 0 };
  data.forEach(item => {
    const parts = item.name.split('.');
    if (parts.length === 1) {
      root._nodes.push(item);
      if (item.kind === 'Param') root._paramCount += item.elements;
      root._bytes += item.bytes;
      return;
    }
    let curr = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      if (!curr._children[p]) {
        curr._children[p] = { _children: {}, _nodes: [], _paramCount: 0, _bytes: 0 };
      }
      if (item.kind === 'Param') curr._children[p]._paramCount += item.elements;
      curr._children[p]._bytes += item.bytes;
      curr = curr._children[p];
    }
    curr._nodes.push(item);
  });
  return root;
}

function renderBranch(prefix, node) {
  let html = '';
  const childKeys = Object.keys(node._children);
  const isRoot = prefix.includes('Root') || prefix.includes('Global');

  if (childKeys.length === 0 && !isRoot) {
    const opInfo = getNodeOpType(prefix);
    return `
      <div class="tb-node-card" onclick="openInspector('${prefix}')" title="Click to inspect ${prefix}" style="margin-bottom: 8px;">
        <div class="tb-node-header">
          <div class="tb-node-title">
            <span class="tb-op-icon ${opInfo.cls}">${opInfo.icon}</span>
            <span>${prefix}</span>
            <span class="tb-type-pill">${opInfo.type}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            ${node._paramCount > 0 ? `<span class="tb-param-chip">⚙️ ${formatNumber(node._paramCount)} params</span>` : ''}
            <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${prefix}')">🔍 Inspect</button>
          </div>
        </div>
      </div>
    `;
  }

  if (node._nodes.length > 0 || childKeys.length > 0) {
    const hasParams = node._paramCount > 0;
    const metaInfo = hasParams ? `${formatNumber(node._paramCount)} params` : `${node._nodes.length} nodes`;
    const titleDisplay = isRoot ? `🌐 ${prefix}` : `📁 ${prefix}`;

    html += `
      <details class="module-group" open data-module="${prefix.toLowerCase()}">
        <summary class="module-header">
          <div class="module-title-box">
            <span class="chevron">▶</span>
            <span class="module-name">${titleDisplay}</span>
          </div>
          <div class="module-meta">
            <span>${metaInfo}</span>
            ${!isRoot ? `<button class="tb-inspect-btn" onclick="event.preventDefault(); event.stopPropagation(); openInspector('${prefix}')">🔍 Inspect</button>` : `<button class="tb-inspect-btn" onclick="event.preventDefault(); event.stopPropagation(); openInspector('root', 'submodules')">🔍 Inspect</button>`}
          </div>
        </summary>
        <div class="module-content">
    `;

    if (isRoot && node._nodes.length > 0) {
      node._nodes.forEach(n => {
        const opInfo = getNodeOpType(n.name);
        html += `
          <div class="tb-node-card" onclick="openInspector('${n.name}')" title="Click to view inputs, parameters & outputs" style="margin-bottom: 8px;">
            <div class="tb-node-header">
              <div class="tb-node-title">
                <span class="tb-op-icon ${opInfo.cls}">${opInfo.icon}</span>
                <span>${n.name}</span>
              </div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <span class="tb-type-pill">${n.kind}</span>
                <span class="tb-shape-chip">${n.shape}</span>
                <button class="tb-inspect-btn" onclick="event.stopPropagation(); openInspector('${n.name}')">🔍 Inspect</button>
              </div>
            </div>
          </div>
        `;
      });
    }

    let renderedChildCount = 0;
    for (const childKey of childKeys) {
      const fullChildName = isRoot ? childKey : `${prefix}.${childKey}`;
      if (renderedChildCount > 0 && !isRoot) {
        html += `
          <div class="tree-flow-connector">
            <div class="flow-line"></div>
            <span class="flow-arrow-text"><span class="arrow-symbol">▼</span> Submodule Sequence</span>
          </div>
        `;
      }
      html += renderBranch(fullChildName, node._children[childKey]);
      renderedChildCount++;
    }

    html += `
        </div>
      </details>
    `;
  }
  return html;
}

function renderTree() {
  const filtered = NODES_DATA.filter(item => {
    const matchesKind = (currentKindFilter === 'all' || item.kind === currentKindFilter);
    const matchesQuery = (currentQuery === '' || 
      item.name.toLowerCase().includes(currentQuery) ||
      (item.shape && item.shape.toLowerCase().includes(currentQuery)) ||
      (item.strategy && item.strategy.toLowerCase().includes(currentQuery))
    );
    return matchesKind && matchesQuery;
  });

  const container = document.getElementById('tree-container');
  if (!container) return;
  if (filtered.length === 0) {
    container.innerHTML = '<div style="text-align: center; padding: 48px; color: var(--text-sub);">No nodes matched the filter criteria.</div>';
    return;
  }

  const hierarchy = buildHierarchy(filtered);
  let outputHtml = '';

  if (hierarchy._nodes.length > 0) {
    outputHtml += renderBranch('(Root / Global Scope)', { _children: {}, _nodes: hierarchy._nodes, _paramCount: hierarchy._paramCount, _bytes: hierarchy._bytes });
  }

  const childKeys = Object.keys(hierarchy._children);
  childKeys.forEach((modKey, idx) => {
    if (idx > 0 || hierarchy._nodes.length > 0) {
      outputHtml += `
        <div class="tree-flow-connector" style="justify-content: center; margin: 4px 0 8px 0;">
          <span class="flow-arrow-text">
            <span class="arrow-symbol">▼</span>
            <span>Default Sequential Data Flow</span>
          </span>
        </div>
      `;
    }
    outputHtml += renderBranch(modKey, hierarchy._children[modKey]);
  });

  container.innerHTML = outputHtml;
}

function setKindFilter(kind, btn) {
  currentKindFilter = kind;
  document.querySelectorAll('.btn-group .btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTree();
}

function filterNodes() {
  currentQuery = document.getElementById('search-input').value.trim().toLowerCase();
  renderTree();
}

function expandAll() {
  document.querySelectorAll('details.module-group').forEach(d => d.open = true);
}

function collapseAll() {
  document.querySelectorAll('details.module-group').forEach(d => d.open = false);
}

// ---------------------------------------------------------------------------
// 8. KPI Stats Initializer
// ---------------------------------------------------------------------------

function initKpis() {
  const sum = (RAW_GRAPH && RAW_GRAPH.summary) ? RAW_GRAPH.summary : {};
  let totalParams = sum.total_params || 0;
  let totalBytes = sum.total_bytes || 0;
  let paramNodes = sum.param_nodes || 0;
  let inputNodes = sum.input_nodes || 0;
  let actNodes = sum.activation_nodes || 0;
  let customInit = sum.custom_init_count || 0;
  let autoGraph = sum.auto_graph_count || 0;
  let totalNodes = sum.total_nodes || NODES_DATA.length;

  if (totalParams === 0) {
    NODES_DATA.forEach(n => {
      if (n.kind === 'Param') totalParams += (n.elements || 0);
      totalBytes += (n.bytes || 0);
      if (n.kind === 'Param') paramNodes++;
      else if (n.kind === 'Input') inputNodes++;
      else actNodes++;
    });
  }

  const elParams = document.getElementById('kpi-total-params');
  if (elParams) elParams.textContent = formatNumber(totalParams);
  const elMem = document.getElementById('kpi-memory-mb');
  if (elMem) elMem.textContent = `~${(totalBytes / (1024 * 1024)).toFixed(2)} MB Memory Footprint`;
  const elParamNodes = document.getElementById('kpi-param-nodes');
  if (elParamNodes) elParamNodes.textContent = formatNumber(paramNodes);
  const elInit = document.getElementById('kpi-init-stats');
  if (elInit) elInit.textContent = `${autoGraph} Auto-Graph / ${customInit} Custom`;
  const elTotalNodes = document.getElementById('kpi-total-nodes');
  if (elTotalNodes) elTotalNodes.textContent = formatNumber(totalNodes);
  const elBreak = document.getElementById('kpi-node-breakdown');
  if (elBreak) elBreak.textContent = `${inputNodes} Inputs · ${actNodes} Activations`;
  const elBuf = document.getElementById('kpi-buffers');
  if (elBuf) elBuf.textContent = formatNumber(totalNodes);

  const btnAll = document.getElementById('btn-filter-all');
  if (btnAll) btnAll.textContent = `All Nodes (${totalNodes})`;
  const btnParam = document.getElementById('btn-filter-param');
  if (btnParam) btnParam.textContent = `Params (${paramNodes})`;
  const btnInput = document.getElementById('btn-filter-input');
  if (btnInput) btnInput.textContent = `Inputs (${inputNodes})`;
  const btnAct = document.getElementById('btn-filter-act');
  if (btnAct) btnAct.textContent = `Activations (${actNodes})`;
}

// ---------------------------------------------------------------------------
// 9. Startup & Embedded Data Hook
// ---------------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
  // Check if embedded data is present in DOM
  const embeddedTag = document.getElementById('znn-model-graph-data');
  if (embeddedTag && embeddedTag.textContent.trim().length > 0) {
    try {
      const data = JSON.parse(embeddedTag.textContent);
      loadGraphData(data);
      return;
    } catch(err) {
      console.warn('Embedded graph parse error:', err);
    }
  }

  // Otherwise, try to auto-fetch default sample json if available
  const samplePaths = [
    'public/sample_model_graph.json',
    '../examples/sample_model_graph.json',
    'examples/sample_model_graph.json'
  ];

  function tryFetch(idx) {
    if (idx >= samplePaths.length) return;
    fetch(samplePaths[idx])
      .then(res => {
        if (!res.ok) throw new Error('Not found');
        return res.json();
      })
      .then(data => loadGraphData(data))
      .catch(() => tryFetch(idx + 1));
  }
  tryFetch(0);

  // Setup drag and drop file loading on body/window
  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  window.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.json') || file.type === 'application/json') {
        const reader = new FileReader();
        reader.onload = (evt) => {
          try {
            const parsed = JSON.parse(evt.target.result);
            loadGraphData(parsed);
          } catch (err) {
            alert('Failed to parse dropped JSON: ' + err.message);
          }
        };
        reader.readAsText(file);
      }
    }
  });
});


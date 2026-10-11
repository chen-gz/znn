/**
 * Antigravity-inspired modern theme and configuration for Mermaid diagrams.
 * Adapts to light (Warm Paper) and dark (Catppuccin Mocha) palettes.
 */

export interface MermaidConfigOptions {
  securityLevel?: "strict" | "loose" | "antiscript" | "sandbox";
}

export function getMermaidConfig(isDark: boolean, options?: MermaidConfigOptions) {
  return {
    startOnLoad: false,
    theme: "base" as const,
    securityLevel: options?.securityLevel || ("strict" as const),
    themeVariables: isDark
      ? {
          darkMode: true,
          fontFamily:
            "'Outfit', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          fontSize: "14px",
          background: "transparent",

          // Flowchart & general nodes
          primaryColor: "#1e1e2e",
          primaryBorderColor: "#45475a",
          primaryTextColor: "#cdd6f4",

          // Secondary nodes
          secondaryColor: "#181825",
          secondaryBorderColor: "#313244",
          secondaryTextColor: "#cdd6f4",

          // Clusters / Subgraphs
          tertiaryColor: "#181825",
          tertiaryBorderColor: "#313244",
          tertiaryTextColor: "#b4befe",
          clusterBkg: "#181825",
          clusterBorder: "#313244",
          titleColor: "#b4befe",

          // Edges & lines
          lineColor: "#89b4fa",
          defaultLinkColor: "#89b4fa",
          edgeLabelBackground: "#1e1e2e",

          // General text
          textColor: "#cdd6f4",
          mainBkg: "#1e1e2e",
          nodeBorder: "#45475a",
          nodeTextColor: "#cdd6f4",

          // Sequence diagrams
          actorBkg: "#1e1e2e",
          actorBorder: "#45475a",
          actorTextColor: "#cdd6f4",
          actorLineColor: "#585b70",
          signalColor: "#89b4fa",
          signalTextColor: "#cdd6f4",
          labelBoxBkgColor: "#181825",
          labelBoxBorderColor: "#313244",
          labelTextColor: "#cdd6f4",
          loopTextColor: "#b4befe",

          // Notes
          noteBkgColor: "#313244",
          noteBorderColor: "#fab387",
          noteTextColor: "#fab387",

          // State & Class
          labelColor: "#cdd6f4",
          altBackground: "#181825",
          classText: "#cdd6f4",

          // Git graph branches
          git0: "#89b4fa",
          git1: "#a6e3a1",
          git2: "#fab387",
          git3: "#cba6f7",
          git4: "#f38ba8",
          git5: "#89dceb",
          git6: "#f9e2af",
          git7: "#b4befe",
        }
      : {
          darkMode: false,
          fontFamily:
            "'Outfit', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          fontSize: "14px",
          background: "transparent",

          // Flowchart & general nodes
          primaryColor: "#ffffff",
          primaryBorderColor: "#dcd6cc",
          primaryTextColor: "#2c3038",

          // Secondary nodes
          secondaryColor: "#f0ede6",
          secondaryBorderColor: "#dcd6cc",
          secondaryTextColor: "#2c3038",

          // Clusters / Subgraphs
          tertiaryColor: "#faf8f5",
          tertiaryBorderColor: "#dcd6cc",
          tertiaryTextColor: "#2c3038",
          clusterBkg: "#faf8f5",
          clusterBorder: "#dcd6cc",
          titleColor: "#20242c",

          // Edges & lines
          lineColor: "#64748b",
          defaultLinkColor: "#64748b",
          edgeLabelBackground: "#ffffff",

          // General text
          textColor: "#2c3038",
          mainBkg: "#ffffff",
          nodeBorder: "#dcd6cc",
          nodeTextColor: "#2c3038",

          // Sequence diagrams
          actorBkg: "#ffffff",
          actorBorder: "#dcd6cc",
          actorTextColor: "#2c3038",
          actorLineColor: "#cbd5e1",
          signalColor: "#64748b",
          signalTextColor: "#2c3038",
          labelBoxBkgColor: "#f0ede6",
          labelBoxBorderColor: "#dcd6cc",
          labelTextColor: "#2c3038",
          loopTextColor: "#20242c",

          // Notes
          noteBkgColor: "#fef3c7",
          noteBorderColor: "#f59e0b",
          noteTextColor: "#92400e",

          // State & Class
          labelColor: "#2c3038",
          altBackground: "#f0ede6",
          classText: "#2c3038",

          // Git graph branches
          git0: "#1a5fb4",
          git1: "#059669",
          git2: "#d97706",
          git3: "#7c3aed",
          git4: "#e11d48",
          git5: "#0284c7",
          git6: "#ca8a04",
          git7: "#4f46e5",
        },
    themeCSS: `
      .node rect, .node polygon, .node circle {
        rx: 8px !important;
        ry: 8px !important;
        stroke-width: 1.5px !important;
        transition: filter 0.2s ease;
      }
      .node:hover rect, .node:hover polygon, .node:hover circle {
        filter: brightness(1.05);
      }
      .cluster rect {
        rx: 12px !important;
        ry: 12px !important;
        stroke-width: 1.25px !important;
        stroke-dasharray: 4 4 !important;
      }
      .cluster .nodeLabel, .cluster text {
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
      }
      .flowchart-link, .edgePath .path {
        stroke-width: 1.75px !important;
      }
      .edgeLabel rect {
        rx: 6px !important;
        ry: 6px !important;
        opacity: 0.95 !important;
      }
      .edgeLabel span {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
      }
      .node foreignObject, .node foreignObject div {
        overflow: visible !important;
      }
      .node, .node p, .node span, .node .nodeLabel {
        font-size: 14px !important;
        line-height: 1.5 !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      .node .label {
        line-height: 1.5 !important;
      }
      .katex {
        font-size: 1.05em !important;
        color: inherit !important;
      }
    `,
    flowchart: {
      htmlLabels: true,
      curve: "basis" as const,
      padding: 18,
      nodeSpacing: 45,
      rankSpacing: 45,
      wrappingWidth: 280,
      subGraphTitleMargin: { top: 8, bottom: 8 },
    },
    sequence: {
      diagramMarginX: 20,
      diagramMarginY: 20,
      actorMargin: 50,
      width: 150,
      height: 45,
      boxMargin: 10,
      boxTextMargin: 5,
      noteMargin: 10,
      messageMargin: 35,
    },
    state: {
      padding: 16,
    },
    class: {
      padding: 16,
    },
    legacyMathML: true,
    forceLegacyMathML: true,
  };
}

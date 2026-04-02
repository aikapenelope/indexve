"use client";

/**
 * PDF source viewer sidebar component.
 *
 * Displays the source chunks used to generate the answer, with:
 * - Document ID and section path
 * - Page reference (clickable in future when PDF viewer is integrated)
 * - Relevance score as a visual bar
 * - Safety level badge (DANGER/WARNING/INFO)
 * - Language indicator (ES/EN)
 * - Text preview of the chunk
 *
 * Reference: docs/PRD.md Section 3.2
 */

interface SourceChunk {
  doc_id: string;
  section_path: string;
  page_ref: string;
  score: number;
  safety_level: string;
  doc_language: string;
  equipment: string;
  text_preview: string;
}

interface SourceViewerProps {
  sources: SourceChunk[];
  isVisible: boolean;
  onToggle: () => void;
}

const SAFETY_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  critico: { bg: "#fecaca", text: "#991b1b", label: "DANGER" },
  precaucion: { bg: "#fef08a", text: "#854d0e", label: "WARNING" },
  informativo: { bg: "#dbeafe", text: "#1e40af", label: "INFO" },
};

function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score * 100));
  const color = pct >= 75 ? "#22c55e" : pct >= 50 ? "#eab308" : "#ef4444";

  return (
    <div
      style={{
        width: "100%",
        height: "4px",
        background: "var(--border)",
        borderRadius: "2px",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${pct}%`,
          height: "100%",
          background: color,
          borderRadius: "2px",
          transition: "width 0.3s ease",
        }}
      />
    </div>
  );
}

function SourceCard({ source, index }: { source: SourceChunk; index: number }) {
  const safety = SAFETY_COLORS[source.safety_level] ?? SAFETY_COLORS.informativo;

  return (
    <div
      style={{
        padding: "0.75rem",
        borderRadius: "0.5rem",
        border: "1px solid var(--border)",
        background: "var(--background)",
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
      }}
    >
      {/* Header: index + safety badge + language */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
        }}
      >
        <span
          style={{
            fontSize: "0.75rem",
            fontWeight: 600,
            color: "var(--muted-foreground)",
          }}
        >
          Fuente {index + 1}
        </span>
        <div style={{ display: "flex", gap: "0.25rem" }}>
          <span
            style={{
              fontSize: "0.625rem",
              padding: "0.125rem 0.375rem",
              borderRadius: "9999px",
              background: safety.bg,
              color: safety.text,
              fontWeight: 700,
            }}
          >
            {safety.label}
          </span>
          <span
            style={{
              fontSize: "0.625rem",
              padding: "0.125rem 0.375rem",
              borderRadius: "9999px",
              background: "var(--muted)",
              color: "var(--muted-foreground)",
              fontWeight: 600,
            }}
          >
            {source.doc_language.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Document info */}
      <div style={{ fontSize: "0.8125rem", fontWeight: 600 }}>
        {source.doc_id}
      </div>
      <div
        style={{
          fontSize: "0.75rem",
          color: "var(--muted-foreground)",
        }}
      >
        {source.section_path && <div>{source.section_path}</div>}
        {source.page_ref && <div>Pagina: {source.page_ref}</div>}
        {source.equipment && <div>Equipo: {source.equipment}</div>}
      </div>

      {/* Score bar */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.6875rem",
            color: "var(--muted-foreground)",
            marginBottom: "0.25rem",
          }}
        >
          <span>Relevancia</span>
          <span>{(source.score * 100).toFixed(0)}%</span>
        </div>
        <ScoreBar score={source.score} />
      </div>

      {/* Text preview */}
      {source.text_preview && (
        <div
          style={{
            fontSize: "0.75rem",
            color: "var(--muted-foreground)",
            lineHeight: 1.4,
            borderTop: "1px solid var(--border)",
            paddingTop: "0.5rem",
            overflow: "hidden",
            display: "-webkit-box",
            WebkitLineClamp: 3,
            WebkitBoxOrient: "vertical",
          }}
        >
          {source.text_preview}
        </div>
      )}
    </div>
  );
}

export function SourceViewer({ sources, isVisible, onToggle }: SourceViewerProps) {
  if (!sources.length) return null;

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={onToggle}
        style={{
          position: "fixed",
          right: isVisible ? "21rem" : "0.5rem",
          top: "50%",
          transform: "translateY(-50%)",
          zIndex: 50,
          padding: "0.5rem",
          borderRadius: "0.5rem 0 0 0.5rem",
          background: "var(--primary)",
          color: "var(--primary-foreground)",
          border: "none",
          cursor: "pointer",
          fontSize: "0.75rem",
          fontWeight: 600,
          writingMode: "vertical-rl",
          transition: "right 0.3s ease",
        }}
      >
        {isVisible ? "Cerrar" : `Fuentes (${sources.length})`}
      </button>

      {/* Sidebar */}
      <aside
        style={{
          position: "fixed",
          right: isVisible ? 0 : "-21rem",
          top: 0,
          bottom: 0,
          width: "20rem",
          background: "var(--muted)",
          borderLeft: "1px solid var(--border)",
          overflowY: "auto",
          padding: "1rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          transition: "right 0.3s ease",
          zIndex: 40,
        }}
      >
        <h3
          style={{
            fontSize: "0.875rem",
            fontWeight: 700,
            color: "var(--foreground)",
            margin: 0,
          }}
        >
          Documentos fuente
        </h3>
        <p
          style={{
            fontSize: "0.75rem",
            color: "var(--muted-foreground)",
            margin: 0,
          }}
        >
          Fragmentos usados para generar la respuesta.
        </p>

        {sources.map((source, i) => (
          <SourceCard key={`${source.doc_id}-${i}`} source={source} index={i} />
        ))}
      </aside>
    </>
  );
}

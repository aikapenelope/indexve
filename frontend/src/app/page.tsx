"use client";

import { useState } from "react";
import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";
import { SourceViewer } from "@/components/source-viewer";
import { manualiqAdapter, lastSources } from "@/lib/manualiq-adapter";

/**
 * ManualIQ main page.
 *
 * Uses assistant-ui LocalRuntime with our custom ChatModelAdapter
 * that calls the FastAPI backend at /query/stream.
 * Includes a source viewer sidebar for PDF citations.
 */
export default function Home() {
  const runtime = useLocalRuntime(manualiqAdapter);
  const [sourcesVisible, setSourcesVisible] = useState(false);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          height: "100dvh",
        }}
      >
        {/* Header */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.75rem 1.5rem",
            borderBottom: "1px solid var(--border)",
            background: "var(--background)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem", fontWeight: 700 }}>
              ManualIQ
            </span>
            <span
              style={{
                fontSize: "0.75rem",
                padding: "0.125rem 0.5rem",
                borderRadius: "9999px",
                background: "var(--accent)",
                color: "var(--primary)",
              }}
            >
              Beta
            </span>
          </div>
          <span
            style={{
              fontSize: "0.8125rem",
              color: "var(--muted-foreground)",
            }}
          >
            Asistente tecnico industrial
          </span>
        </header>

        {/* Chat */}
        <main style={{ flex: 1, overflow: "hidden" }}>
          <Thread />
        </main>

        {/* Source viewer sidebar */}
        <SourceViewer
          sources={lastSources}
          isVisible={sourcesVisible}
          onToggle={() => setSourcesVisible(!sourcesVisible)}
        />

        {/* Footer disclaimer */}
        <footer
          style={{
            padding: "0.5rem 1rem",
            textAlign: "center",
            fontSize: "0.75rem",
            color: "var(--muted-foreground)",
            borderTop: "1px solid var(--border)",
          }}
        >
          Verificar siempre con el supervisor antes de ejecutar procedimientos
          criticos de seguridad.
        </footer>
      </div>
    </AssistantRuntimeProvider>
  );
}

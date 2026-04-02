"use client";

import {
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { SendHorizontal } from "lucide-react";

/**
 * ManualIQ chat thread component built with assistant-ui primitives.
 *
 * Renders:
 * - Message list with markdown support (for citations, bold safety warnings)
 * - Composer with send button
 * - User and assistant message styling
 */
export function Thread() {
  return (
    <ThreadPrimitive.Root
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        maxWidth: "48rem",
        margin: "0 auto",
      }}
    >
      <ThreadPrimitive.Viewport
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "1rem",
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
        }}
      >
        <ThreadPrimitive.Empty>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              gap: "0.5rem",
              color: "var(--muted-foreground)",
              textAlign: "center",
              padding: "2rem",
            }}
          >
            <h2 style={{ fontSize: "1.5rem", fontWeight: 600 }}>ManualIQ</h2>
            <p>Asistente tecnico para manuales industriales.</p>
            <p style={{ fontSize: "0.875rem" }}>
              Pregunte en espanol sobre cualquier equipo. Las respuestas incluyen
              citas exactas del manual fuente.
            </p>
          </div>
        </ThreadPrimitive.Empty>

        <ThreadPrimitive.Messages
          components={{
            UserMessage,
            AssistantMessage,
          }}
        />
      </ThreadPrimitive.Viewport>

      <Composer />
    </ThreadPrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root
      style={{
        display: "flex",
        justifyContent: "flex-end",
      }}
    >
      <div
        style={{
          maxWidth: "80%",
          padding: "0.75rem 1rem",
          borderRadius: "1rem 1rem 0.25rem 1rem",
          background: "var(--primary)",
          color: "var(--primary-foreground)",
          fontSize: "0.9375rem",
          lineHeight: 1.5,
        }}
      >
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root
      style={{
        display: "flex",
        justifyContent: "flex-start",
      }}
    >
      <div
        style={{
          maxWidth: "85%",
          padding: "0.75rem 1rem",
          borderRadius: "1rem 1rem 1rem 0.25rem",
          background: "var(--muted)",
          fontSize: "0.9375rem",
          lineHeight: 1.6,
        }}
      >
        <MessagePrimitive.Content
          components={{
            Text: MarkdownTextPrimitive as never,
          }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

function Composer() {
  return (
    <ComposerPrimitive.Root
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: "0.5rem",
        padding: "0.75rem 1rem",
        borderTop: "1px solid var(--border)",
        background: "var(--background)",
      }}
    >
      <ComposerPrimitive.Input
        placeholder="Pregunte sobre un manual tecnico..."
        style={{
          flex: 1,
          padding: "0.625rem 0.875rem",
          borderRadius: "0.75rem",
          border: "1px solid var(--border)",
          background: "var(--muted)",
          color: "var(--foreground)",
          fontSize: "0.9375rem",
          outline: "none",
          resize: "none",
        }}
      />
      <ComposerPrimitive.Send
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: "2.5rem",
          height: "2.5rem",
          borderRadius: "0.75rem",
          background: "var(--primary)",
          color: "var(--primary-foreground)",
          border: "none",
          cursor: "pointer",
          flexShrink: 0,
        }}
      >
        <SendHorizontal size={18} />
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
}

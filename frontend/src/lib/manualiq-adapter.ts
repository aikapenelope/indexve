"use client";

import type { ChatModelAdapter, ChatModelRunResult } from "@assistant-ui/react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

interface StreamEvent {
  type: "text" | "sources" | "error";
  content?: string;
  sources?: SourceChunk[];
}

/**
 * Mutable array holding the sources from the last query.
 * Read by the SourceViewer component in page.tsx.
 */
export let lastSources: SourceChunk[] = [];

export const manualiqAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }): AsyncGenerator<ChatModelRunResult> {
    const lastMessage = messages[messages.length - 1];
    const query =
      lastMessage?.role === "user"
        ? lastMessage.content
            .filter((part) => part.type === "text")
            .map((part) => part.text)
            .join(" ")
        : "";

    if (!query.trim()) {
      yield {
        content: [
          {
            type: "text" as const,
            text: "Por favor, escriba una pregunta sobre los manuales tecnicos.",
          },
        ],
      };
      return;
    }

    // Reset sources for this query.
    lastSources = [];

    const response = await fetch(`${API_URL}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-tenant-id": "dev_tenant",
        "x-user-id": "dev_user",
      },
      body: JSON.stringify({ query }),
      signal: abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      yield {
        content: [
          {
            type: "text" as const,
            text: `Error del servidor (${response.status}): ${errorText}`,
          },
        ],
      };
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      yield {
        content: [{ type: "text" as const, text: "Error: No stream available" }],
      };
      return;
    }

    const decoder = new TextDecoder();
    let accumulated = "";
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();

          if (data === "[DONE]") break;

          try {
            const event: StreamEvent = JSON.parse(data);

            if (event.type === "text" && event.content) {
              accumulated += event.content;
              yield {
                content: [{ type: "text" as const, text: accumulated }],
              };
            }

            if (event.type === "sources" && event.sources) {
              lastSources = event.sources;
            }
          } catch {
            // Skip malformed JSON lines.
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    if (accumulated) {
      yield {
        content: [{ type: "text" as const, text: accumulated }],
      };
    }
  },
};

"use client";

import type { ChatModelAdapter, ChatModelRunResult } from "@assistant-ui/react";

/**
 * API base URL for the ManualIQ FastAPI backend.
 * In production, this comes from NEXT_PUBLIC_API_URL env var.
 * In development, defaults to localhost:8000.
 */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * SSE event from the /query/stream endpoint.
 */
interface StreamEvent {
  type: "text" | "sources" | "error";
  content?: string;
  sources?: Array<{
    doc_id: string;
    section_path: string;
    page_ref: string;
    score: number;
    safety_level: string;
  }>;
}

/**
 * ChatModelAdapter that connects assistant-ui to the ManualIQ FastAPI backend.
 *
 * Uses the streaming endpoint (/query/stream) for token-by-token response.
 * First token arrives in ~500ms, full response streams over 2-4 seconds.
 *
 * The tenant_id and user_id are injected via headers (server-side in
 * production via Clerk middleware; dev mode uses hardcoded values).
 */
export const manualiqAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }): AsyncGenerator<ChatModelRunResult> {
    // Extract the last user message as the query.
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

    const response = await fetch(`${API_URL}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Dev mode headers. In production, Clerk middleware injects these.
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

    // Read the SSE stream.
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

        // Process complete SSE lines.
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
            // Sources events are received but not rendered as text.
            // The frontend can use them for the PDF viewer sidebar.
          } catch {
            // Skip malformed JSON lines.
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    // Final yield with complete text.
    if (accumulated) {
      yield {
        content: [{ type: "text" as const, text: accumulated }],
      };
    }
  },
};

"use client";

import type { ChatModelAdapter } from "@assistant-ui/react";

/**
 * API base URL for the ManualIQ FastAPI backend.
 * In production, this comes from NEXT_PUBLIC_API_URL env var.
 * In development, defaults to localhost:8000.
 */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Response shape from POST /query on the FastAPI backend.
 */
interface QueryResponse {
  answer: string;
  confidence: string;
  score: number;
  chunks_used: number;
  chunks_retrieved: number;
  was_fallback: boolean;
  intent: string;
}

/**
 * ChatModelAdapter that connects assistant-ui to the ManualIQ FastAPI backend.
 *
 * Uses LocalRuntime pattern: assistant-ui manages state, we provide the
 * model adapter that calls our API and returns the response.
 *
 * The tenant_id and user_id are injected via headers (server-side in
 * production via Clerk middleware; dev mode uses hardcoded values).
 */
export const manualiqAdapter: ChatModelAdapter = {
  async run({ messages, abortSignal }) {
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
      return {
        content: [
          {
            type: "text" as const,
            text: "Por favor, escriba una pregunta sobre los manuales tecnicos.",
          },
        ],
      };
    }

    const response = await fetch(`${API_URL}/query`, {
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
      return {
        content: [
          {
            type: "text" as const,
            text: `Error del servidor (${response.status}): ${errorText}`,
          },
        ],
      };
    }

    const data: QueryResponse = await response.json();

    // Build the response with metadata for the UI.
    let text = data.answer;

    // Append confidence badge if not high.
    if (data.confidence !== "high" && data.confidence !== "none") {
      text = `> Confianza: ${data.confidence} (score: ${data.score.toFixed(2)})\n\n${text}`;
    }

    return {
      content: [{ type: "text" as const, text }],
    };
  },
};

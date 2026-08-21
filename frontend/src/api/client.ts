const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface ProviderStatus {
  configured: boolean;
  reachable: boolean | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  providers: Record<string, ProviderStatus>;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok && response.status !== 503) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}

export interface GuardrailInfo {
  stage: string;
  reason_code: string;
  message: string;
  score: number | null;
}

export interface SourceInfo {
  chunk_id: string;
  text: string;
}

export interface TimingsInfo {
  transcribe_ms: number | null;
  retrieve_ms: number | null;
  generate_ms: number | null;
  total_ms: number;
}

export interface QueryResponse {
  request_id: string;
  status: "ok" | "rejected" | "error";
  transcript: string | null;
  query_language: string | null;
  answer: string | null;
  confidence: number | null;
  sources: SourceInfo[];
  reranked: boolean | null;
  guardrail: GuardrailInfo | null;
  error: string | null;
  timings: TimingsInfo;
}

export async function queryAudio(
  audio: Blob,
  languageHint = "hi-IN",
  useReranker = false,
): Promise<QueryResponse> {
  const formData = new FormData();
  formData.append("audio", audio, "recording.webm");
  formData.append("language_hint", languageHint);
  formData.append("use_reranker", String(useReranker));

  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    body: formData,
  });

  if (response.status === 400) {
    const body = (await response.json().catch(() => null)) as { error?: string } | null;
    throw new Error(body?.error ?? `Request rejected with status 400`);
  }

  return (await response.json()) as QueryResponse;
}

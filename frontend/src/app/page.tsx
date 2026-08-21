"use client";

import { useRef, useState } from "react";
import { queryAudio, type QueryResponse } from "@/api/client";

type Phase = "idle" | "recording" | "submitting" | "done" | "error";

const BAR_COUNT = 5;

const SOURCE_PREVIEW_CHARS = 150;

function SourceCard({ chunkId, text }: { chunkId: string; text: string }) {
  const isLong = text.length > SOURCE_PREVIEW_CHARS;
  const preview = isLong ? `${text.slice(0, SOURCE_PREVIEW_CHARS)}…` : text;

  return (
    <div className="rounded-md border-l-2 border-[#3a3a4a] bg-[#1b1b24] px-3 py-2">
      {isLong ? (
        <details>
          <summary className="cursor-pointer text-xs leading-relaxed text-[#8b889a] marker:content-none">
            {preview}
          </summary>
          <p className="mt-1 text-xs leading-relaxed text-[#8b889a]">{text}</p>
        </details>
      ) : (
        <p className="text-xs leading-relaxed text-[#8b889a]">{preview}</p>
      )}
      <p className="mt-1 text-[10px] tracking-wide text-[#5a5868] uppercase">{chunkId}</p>
    </div>
  );
}

function MicIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" />
      <path
        d="M5 11a7 7 0 0 0 14 0M12 18v3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [levels, setLevels] = useState<number[]>(Array(BAR_COUNT).fill(0.1));
  const [useReranker, setUseReranker] = useState(false);
  // One reading per toggle state, so the latency tradeoff (the point of
  // this feature) stays visible on screen after trying both.
  const [comparisonMs, setComparisonMs] = useState<{ off: number | null; on: number | null }>({
    off: null,
    on: null,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);

  function startWaveform() {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const step = Math.max(1, Math.floor(data.length / BAR_COUNT));

    const tick = () => {
      analyser.getByteFrequencyData(data);
      setLevels(
        Array.from({ length: BAR_COUNT }, (_, i) => Math.max(0.1, (data[i * step] ?? 0) / 255)),
      );
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  function stopWaveform() {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setLevels(Array(BAR_COUNT).fill(0.1));
  }

  async function startRecording() {
    setErrorMessage(null);
    setResult(null);

    if (typeof MediaRecorder === "undefined") {
      setPhase("error");
      setErrorMessage("This browser doesn't support audio recording.");
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setPhase("error");
      setErrorMessage(
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone access was denied. Allow it and try again."
          : `Couldn't access the microphone: ${err instanceof Error ? err.message : String(err)}`,
      );
      return;
    }

    streamRef.current = stream;
    chunksRef.current = [];

    try {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new Ctx();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser); // not connected to destination, no echo
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      startWaveform();
    } catch {
      // Waveform is a bonus signal, recording still works without it.
    }

    const recorder = new MediaRecorder(stream);
    mediaRecorderRef.current = recorder;
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => void submitRecording();
    recorder.start();
    setPhase("recording");
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    stopWaveform();
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    analyserRef.current = null;
  }

  async function submitRecording() {
    setPhase("submitting");
    const audioBlob = new Blob(chunksRef.current, {
      type: mediaRecorderRef.current?.mimeType || "audio/webm",
    });
    const rerankerModeForThisRequest = useReranker;

    try {
      const response = await queryAudio(audioBlob, "hi-IN", rerankerModeForThisRequest);
      setResult(response);
      setComparisonMs((prev) => ({
        ...prev,
        [rerankerModeForThisRequest ? "on" : "off"]: response.timings.total_ms,
      }));
      setPhase("done");
    } catch (err) {
      setPhase("error");
      setErrorMessage(
        err instanceof TypeError
          ? "Couldn't reach the backend. Is it running?"
          : `Request failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  function handleClick() {
    if (phase === "recording") stopRecording();
    else void startRecording();
  }

  const label: Record<Phase, string> = {
    idle: "Listening",
    recording: "Listening...",
    submitting: "Thinking...",
    done: "Listening",
    error: "Listening",
  };

  return (
    <main className="flex min-h-screen justify-center bg-[#15151c] px-6 py-16 text-[#f2f0ea]">
      <div className="flex w-full max-w-sm flex-col">
        <h1 className="text-lg font-medium tracking-tight text-[#f2f0ea]">Voice RAG</h1>

        <div className="mt-20 flex flex-col items-center">
          <button
            onClick={handleClick}
            disabled={phase === "submitting"}
            aria-label={phase === "recording" ? "Stop recording" : "Start recording"}
            className={`flex h-24 w-24 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed ${
              phase === "recording"
                ? "bg-[#e8a33d] text-[#15151c]"
                : "bg-[#232330] text-[#e8a33d] hover:bg-[#2b2b3a]"
            }`}
          >
            {phase === "recording" ? (
              <div className="flex h-5 items-end gap-1" aria-hidden>
                {levels.map((v, i) => (
                  <span
                    key={i}
                    className="w-1 rounded-full bg-[#15151c]"
                    style={{ height: `${6 + v * 20}px` }}
                  />
                ))}
              </div>
            ) : phase === "submitting" ? (
              <span className="h-3 w-3 animate-pulse rounded-full bg-[#e8a33d]" />
            ) : (
              <MicIcon />
            )}
          </button>

          <p className="mt-4 text-sm text-[#a3a0b0]">{label[phase]}</p>

          <button
            type="button"
            role="switch"
            aria-checked={useReranker}
            onClick={() => setUseReranker((v) => !v)}
            disabled={phase === "recording" || phase === "submitting"}
            className="mt-6 flex items-center gap-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span className="text-xs text-[#a3a0b0]">सटीकता मोड</span>
            <span
              className={`relative h-5 w-9 rounded-full transition-colors ${
                useReranker ? "bg-[#e8a33d]" : "bg-[#2b2b3a]"
              }`}
            >
              <span
                className={`absolute top-0.5 h-4 w-4 rounded-full bg-[#f2f0ea] transition-transform ${
                  useReranker ? "translate-x-4" : "translate-x-0.5"
                }`}
              />
            </span>
            <span className="text-xs text-[#a3a0b0]">{useReranker ? "चालू" : "बंद"}</span>
          </button>
        </div>

        {phase === "error" && errorMessage && (
          <p className="mt-10 rounded-md bg-[#232330] p-3 text-xs leading-relaxed text-[#a3a0b0]">
            {errorMessage}
          </p>
        )}

        {result && (
          <div className="mt-10 flex flex-col gap-5">
            {result.transcript && (
              <p className="text-sm leading-relaxed text-[#a3a0b0]">{result.transcript}</p>
            )}

            {result.status === "ok" && (
              <div className="border-l-2 border-[#6fae7f] pl-4">
                <p className="text-base leading-relaxed text-[#f2f0ea]">{result.answer}</p>

                {result.sources.length > 0 && (
                  <div className="mt-3 flex flex-col gap-2">
                    <p className="text-xs text-[#6fae7f]">स्रोत</p>
                    {result.sources.map((s) => (
                      <SourceCard key={s.chunk_id} chunkId={s.chunk_id} text={s.text} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {result.status === "rejected" && result.guardrail && (
              <div className="border-l-2 border-[#a3a0b0] pl-4">
                <p className="text-sm text-[#f2f0ea]">
                  {result.guardrail.message}
                  {result.guardrail.score !== null && (
                    <span className="text-[#6b6878]"> (स्कोर: {result.guardrail.score.toFixed(2)})</span>
                  )}
                </p>
                <p className="mt-1 text-xs text-[#6b6878]">
                  {result.guardrail.stage} · {result.guardrail.reason_code}
                </p>
              </div>
            )}

            {result.status === "error" && (
              <div className="border-l-2 border-[#c96a5a] pl-4">
                <p className="text-sm text-[#f2f0ea]">{result.error ?? "Something went wrong."}</p>
              </div>
            )}

            {result.timings.total_ms > 0 && (
              <div className="flex flex-col gap-0.5">
                <p className="text-[11px] text-[#5a5868]">
                  {Math.round(result.timings.total_ms)}ms में जवाब मिला
                  {result.reranked !== null && (
                    <span> · रीरैंक {result.reranked ? "हुआ" : "नहीं हुआ"}</span>
                  )}
                </p>
                {comparisonMs.off !== null && comparisonMs.on !== null && (
                  <p className="text-[11px] text-[#5a5868]">
                    तुलना: बंद {Math.round(comparisonMs.off)}ms, चालू {Math.round(comparisonMs.on)}ms
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
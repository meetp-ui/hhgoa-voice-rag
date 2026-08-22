"use client";

import { useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { queryAudio, queryText as apiQueryText, type QueryResponse } from "@/lib/api/client";

type Phase = "idle" | "recording" | "submitting" | "done" | "error";

const BAR_COUNT = 12;
const SOURCE_PREVIEW_CHARS = 160;
const SILENCE_VOLUME_THRESHOLD = 0.04;

const LANGUAGES = [
  { code: "auto", label: "+ Auto-detect", hint: "" },
  { code: "en",   label: "English",       hint: "en-US" },
  { code: "hi",   label: "हिन्दी",         hint: "hi-IN" },
] as const;

type LangCode = (typeof LANGUAGES)[number]["code"];

const SAMPLE_QUESTIONS = [
  { lang: "English", text: "what is a corporation" },
  { lang: "English", text: "are corn meal and corn flour the same" },
  { lang: "हिन्दी",   text: "जलवायु मौसम का अध्यन" },
  { lang: "हिन्दी",   text: "कोरोना वायरस क्या है" },
];

const SILENCE_MARKS = [
  { ms: 150,  label: "SNAPPY" },
  { ms: 300,  label: "DEFAULT" },
  { ms: 650,  label: "RELAXED" },
  { ms: 1400, label: "THINKING ALOUD" },
];

function MicIcon({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
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

function SourceCard({ chunkId, text }: { chunkId: string; text: string }) {
  const isLong = text.length > SOURCE_PREVIEW_CHARS;
  const preview = isLong ? `${text.slice(0, SOURCE_PREVIEW_CHARS)}…` : text;
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-[#D0C8A8] bg-[#F5F1E3] px-4 py-3"
    >
      {isLong ? (
        <details>
          <summary className="cursor-pointer text-xs leading-relaxed text-[#4A4A3A] marker:content-none">
            {preview}
          </summary>
          <p className="mt-1 text-xs leading-relaxed text-[#4A4A3A]">{text}</p>
        </details>
      ) : (
        <p className="text-xs leading-relaxed text-[#4A4A3A]">{preview}</p>
      )}
      <p className="mt-2 text-[10px] tracking-widest text-[#8A826A] uppercase font-medium">
        {chunkId}
      </p>
    </motion.div>
  );
}

function TimerDisplay({ ms, budget = 200 }: { ms: number; budget?: number }) {
  const intPart = Math.floor(ms);
  const fracPart = Math.round((ms % 1) * 10);
  const overBudget = ms > budget;
  const color = overBudget ? "#8B2E1E" : "#1C3A20";
  return (
    <div className="flex items-baseline gap-0.5 tabular-nums" style={{ fontFamily: 'var(--font-fraunces), Georgia, serif' }}>
      <span style={{ color }} className="text-5xl font-bold tracking-tight leading-none">
        {intPart}
      </span>
      <span style={{ color }} className="text-3xl font-medium leading-none pb-0.5">
        .
      </span>
      <span style={{ color }} className="text-5xl font-bold tracking-tight leading-none">
        {fracPart}
      </span>
      <span className="text-base text-[#4A7A52] font-medium ml-1">ms</span>
    </div>
  );
}

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [transcript, setTranscript] = useState<string>("");

  const [levels, setLevels] = useState<number[]>(Array(BAR_COUNT).fill(0.1));
  const [selectedLang, setSelectedLang] = useState<LangCode>("auto");
  const [useReranker, setUseReranker] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [silenceMs, setSilenceMs] = useState(300);
  const silenceMsRef = useRef(300);

  const [comparisonMs, setComparisonMs] = useState<{ off: number | null; on: number | null }>({
    off: null,
    on: null,
  });
  const mediaRecorderRef  = useRef<MediaRecorder | null>(null);
  const chunksRef         = useRef<Blob[]>([]);
  const streamRef         = useRef<MediaStream | null>(null);
  const audioCtxRef       = useRef<AudioContext | null>(null);
  const analyserRef       = useRef<AnalyserNode | null>(null);
  const rafRef            = useRef<number | null>(null);
  const silenceStartRef   = useRef<number | null>(null);
  const stopCalledRef     = useRef(false);

  const startWaveform = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const step = Math.max(1, Math.floor(data.length / BAR_COUNT));
    silenceStartRef.current = null;
    stopCalledRef.current = false;

    const tick = () => {
      if (stopCalledRef.current) return;
      analyser.getByteFrequencyData(data);

      const avgVol = data.reduce((a, b) => a + b, 0) / data.length / 255;
      if (avgVol < SILENCE_VOLUME_THRESHOLD) {
        if (silenceStartRef.current === null) {
          silenceStartRef.current = Date.now();
        } else if (Date.now() - silenceStartRef.current >= silenceMsRef.current) {
          silenceStartRef.current = null;
          stopCalledRef.current = true;
          stopRecording();
          return;
        }
      } else {
        silenceStartRef.current = null;
      }

      setLevels(
        Array.from({ length: BAR_COUNT }, (_, i) => Math.max(0.1, (data[i * step] ?? 0) / 255)),
      );
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  function stopWaveform() {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setLevels(Array(BAR_COUNT).fill(0.1));
  }


  async function startRecording() {
    setErrorMessage(null);
    setResult(null);
    setTranscript("");

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
      const source   = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      audioCtxRef.current  = audioCtx;
      analyserRef.current  = analyser;
      startWaveform();
    } catch {
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
    analyserRef.current  = null;
  }

  async function submitRecording() {
    setPhase("submitting");
    const audioBlob = new Blob(chunksRef.current, {
      type: mediaRecorderRef.current?.mimeType ?? "audio/webm",
    });
    const rerankerMode = useReranker;
    const langHint = LANGUAGES.find((l) => l.code === selectedLang)?.hint ?? "";

    try {
      const response = await queryAudio(audioBlob, langHint || "hi-IN", rerankerMode);
      setResult(response);
      if (response.transcript) setTranscript(response.transcript);
      setComparisonMs((prev) => ({
        ...prev,
        [rerankerMode ? "on" : "off"]: response.timings.total_ms,
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

  async function handleTextQuery(queryStr: string) {
    if (!queryStr.trim()) return;
    setErrorMessage(null);
    setResult(null);
    setTranscript(queryStr.trim());
    setPhase("submitting");
    const rerankerMode = useReranker;
    const langHint = LANGUAGES.find((l) => l.code === selectedLang)?.hint;

    try {
      const response = await apiQueryText(queryStr.trim(), langHint, rerankerMode);
      setResult(response);
      setComparisonMs((prev) => ({
        ...prev,
        [rerankerMode ? "on" : "off"]: response.timings.total_ms,
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

  function handleMicClick() {
    if (phase === "recording") stopRecording();
    else if (phase !== "submitting") void startRecording();
  }

  function handleSampleClick(text: string) {
    setTextInput(text);
  }

  function handleAskSubmit(e: React.FormEvent) {
    e.preventDefault();
    void handleTextQuery(textInput);
    setTextInput("");
  }

  const stage1Ms    = result?.timings.retrieve_ms  ?? 0;
  const transcribeMs = result?.timings.transcribe_ms ?? null;
  const generateMs   = result?.timings.generate_ms  ?? null;
  const totalMs      = result?.timings.total_ms      ?? 0;

  const silencePercent = ((silenceMs - 100) / (1500 - 100)) * 100;

  return (
    <div className="min-h-screen bg-[#EEE9D1] text-[#1A1A0E]">

      {/* Decorative vertical strips (large screens only) */}
      <div
        className="fixed left-0 top-0 h-full w-7 hidden xl:flex items-center justify-center pointer-events-none z-10"
        aria-hidden
      >
        <span
          className="text-[9px] tracking-[0.3em] text-[#A8A090] uppercase"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          The Ocean at Your Doorstep · HH Goa 2025
        </span>
      </div>
      <div
        className="fixed right-0 top-0 h-full w-7 hidden xl:flex items-center justify-center pointer-events-none z-10"
        aria-hidden
      >
        <span
          className="text-[9px] tracking-[0.3em] text-[#A8A090] uppercase"
          style={{ writingMode: "vertical-rl" }}
        >
          250,000 Passages · Two Languages · 30 ms Budget
        </span>
      </div>

      <main className="mx-auto max-w-6xl px-4 sm:px-6 xl:px-12 py-8 sm:py-10">

        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 sm:gap-4 mb-8">
          <div className="min-w-0">
            <p className="text-[11px] tracking-[0.2em] uppercase font-semibold text-[#1C3A20] mb-1">
              Less Noise. More Signal.
            </p>
            <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold leading-tight" style={{ fontFamily: 'var(--font-fraunces), Georgia, serif' }}>
              Voice RAG{" "}
              <span className="text-[#4A7A52] font-light">over MSMARCO-XI</span>
            </h1>
            <p className="mt-2 sm:mt-3 text-sm text-[#4A4A3A] leading-relaxed max-w-xl">
              Ask a question out loud, in any of two languages, and it searches 250,000 passages
              while you are still speaking. It answers only from what it actually found — cited,
              in under 200 ms — or it tells you the corpus cannot support an answer.
            </p>
          </div>
          <Link
            href="/benchmarks"
            className="self-start flex-shrink-0 rounded-full border border-[#1C3A20] px-4 py-2 sm:py-1.5 text-sm text-[#1C3A20] hover:bg-[#1C3A20] hover:text-[#EEE9D1] transition-colors whitespace-nowrap"
          >
            Benchmarks
          </Link>
        </div>

        <div className="flex flex-wrap gap-2 mb-6 sm:mb-8">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              onClick={() => setSelectedLang(lang.code)}
              className={`rounded-full px-3 sm:px-4 py-2 sm:py-1.5 text-sm font-medium transition-all border ${
                selectedLang === lang.code
                  ? lang.code === "auto"
                    ? "bg-[#D4CA00] text-[#1A1A0E] border-[#D4CA00] shadow-sm"
                    : "bg-[#1C3A20] text-[#EEE9D1] border-[#1C3A20] shadow-sm"
                  : "bg-transparent text-[#1A1A0E] border-[#D0C8A8] hover:border-[#1C3A20]"
              }`}
            >
              {lang.label}
            </button>
          ))}
        </div>

        <div className="flex flex-col lg:flex-row gap-6">

          <div className="flex-1 min-w-0">
            <div className="flex gap-4 sm:gap-5 items-start mb-6 sm:mb-7">
              <div className="relative flex-shrink-0">
                {phase === "recording" && (
                  <motion.span
                    className="absolute inset-0 rounded-full bg-[#D4CA00]"
                    animate={{ scale: [1, 1.35, 1.35], opacity: [0.6, 0, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
                    aria-hidden
                  />
                )}
                <motion.button
                  onClick={handleMicClick}
                  disabled={phase === "submitting"}
                  aria-label={phase === "recording" ? "Stop recording" : "Start recording"}
                  whileTap={{ scale: 0.93 }}
                  className={`relative flex h-16 w-16 sm:h-20 sm:w-20 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed ${
                    phase === "recording"
                      ? "bg-[#D4CA00] text-[#1A1A0E]"
                      : phase === "submitting"
                        ? "bg-[#E4DFCA] text-[#8A826A] cursor-not-allowed"
                        : "bg-[#D4CA00] text-[#1A1A0E] hover:bg-[#EDE200]"
                  }`}
                >
                  {phase === "recording" ? (
                    <div className="flex h-5 sm:h-6 items-end gap-[2px] sm:gap-[3px]" aria-hidden>
                      {levels.map((v, i) => (
                        <motion.span
                          key={i}
                          animate={{ height: `${4 + v * 14}px` }}
                          transition={{ duration: 0.05, ease: "linear" }}
                          className="w-[2px] sm:w-[3px] rounded-full bg-[#1A1A0E]"
                          style={{ height: `${4 + v * 14}px` }}
                        />
                      ))}
                    </div>
                  ) : phase === "submitting" ? (
                    <motion.span
                      animate={{ opacity: [0.4, 1, 0.4] }}
                      transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                      className="h-3 w-3 rounded-full bg-[#1C3A20]"
                    />
                  ) : (
                    <MicIcon size={24} />
                  )}
                </motion.button>
              </div>

              <div className="flex-1 pt-2">
                <p className="text-[10px] tracking-[0.18em] uppercase text-[#7A7868] mb-1 font-semibold">
                  Transcript
                </p>
                <p
                  className={`text-base leading-relaxed ${
                    transcript ? "text-[#1A1A0E]" : "text-[#A8A090] italic"
                  }`}
                >
                  {transcript || "Tap the mic, or type below"}
                </p>
              </div>
            </div>

            <form onSubmit={handleAskSubmit} className="flex gap-2 mb-6">
              <input
                type="text"
                placeholder="…or type a question"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                disabled={phase === "recording" || phase === "submitting"}
                className="flex-1 rounded-lg border border-[#D0C8A8] bg-[#F5F1E3] px-4 py-3 sm:py-2.5 text-[16px] sm:text-sm text-[#1A1A0E] placeholder:text-[#A8A090] focus:outline-none focus:border-[#1C3A20] transition-colors disabled:opacity-50"
              />
              <motion.button
                type="submit"
                disabled={!textInput.trim() || phase === "recording" || phase === "submitting"}
                whileTap={{ scale: 0.95 }}
                className="rounded-lg bg-[#D4CA00] px-5 py-3 sm:py-2.5 text-sm font-semibold text-[#1A1A0E] hover:bg-[#EDE200] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Ask
              </motion.button>
            </form>

            <div className="mb-8">
              <p className="text-[10px] tracking-[0.18em] uppercase text-[#7A7868] font-semibold mb-3">
                In this corpus · 5,000 topics from MS Marco
              </p>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q.text}
                    onClick={() => handleSampleClick(q.text)}
                    className="rounded-full border border-[#D0C8A8] bg-[#F5F1E3] px-3 py-1.5 text-xs text-[#4A4A3A] hover:border-[#1C3A20] hover:text-[#1C3A20] transition-colors"
                  >
                    <span className="text-[#7A7868]">{q.lang}</span>
                    <span className="text-[#C0B898] mx-1">·</span>
                    {q.text.length > 40 ? q.text.slice(0, 40) + "…" : q.text}
                  </button>
                ))}
              </div>
              <p className="mt-3 text-[11px] text-[#7A7868] leading-relaxed">
                Ask something outside those 5,000 topics — "what is a country" — and it will
                decline rather than guess. That is the intended behaviour.
              </p>
            </div>

            <div className="flex items-center gap-3 mb-5 sm:mb-6">
              <span className="text-xs text-[#7A7868]">Accuracy mode (reranker)</span>
              <button
                type="button"
                role="switch"
                aria-checked={useReranker}
                onClick={() => setUseReranker((v) => !v)}
                disabled={phase === "recording" || phase === "submitting"}
                className="relative h-5 w-9 rounded-full transition-colors disabled:opacity-40 rotate-180 disabled:cursor-not-allowed"
                style={{ background: useReranker ? "#1C3A20" : "#D0C8A8" }}
              >
                <span
                  className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${
                    useReranker ? "-translate-x-4" : "translate-x-0.5"
                  }`}
                />
              </button>
              <span className="text-xs text-[#7A7868]">{useReranker ? "On" : "Off"}</span>
            </div>

            <AnimatePresence>
              {phase === "error" && errorMessage && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="rounded-lg border border-[#C8A090] bg-[#F5E8E5] px-4 py-3 text-sm text-[#8B2E1E]"
                >
                  {errorMessage}
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence mode="wait">
              {result && (
                <motion.div
                  key={result.request_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mt-6 flex flex-col gap-4"
                >
                  {result.transcript && result.transcript !== transcript && (
                    <p className="text-sm text-[#7A7868] italic">
                      &ldquo;{result.transcript}&rdquo;
                    </p>
                  )}

                  {result.query_language && (
                    <p className="text-[10px] tracking-widest uppercase text-[#7A7868] font-semibold">
                      Detected: {result.query_language}
                      {result.reranked !== null && (
                        <span> · Reranked: {result.reranked ? "yes" : "no"}</span>
                      )}
                    </p>
                  )}

                  {result.status === "ok" && (
                    <div className="border-l-2 border-[#1C3A20] pl-4">
                      <p className="text-base leading-relaxed text-[#1A1A0E]">{result.answer}</p>
                      {result.confidence !== null && (
                        <p className="mt-1 text-[11px] text-[#7A7868]">
                          Confidence: {(result.confidence * 100).toFixed(0)}%
                        </p>
                      )}
                      {result.sources.length > 0 && (
                        <div className="mt-4 flex flex-col gap-2">
                          <p className="text-[10px] tracking-widest uppercase text-[#4A7A52] font-semibold">
                            Sources
                          </p>
                          {result.sources.map((s) => (
                            <SourceCard key={s.chunk_id} chunkId={s.chunk_id} text={s.text} />
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {result.status === "rejected" && result.guardrail && (
                    <div className="border-l-2 border-[#D0C8A8] pl-4">
                      <p className="text-sm text-[#1A1A0E] leading-relaxed">
                        {result.guardrail.message}
                        {result.guardrail.score !== null && (
                          <span className="text-[#7A7868]">
                            {" "}(score: {result.guardrail.score.toFixed(2)})
                          </span>
                        )}
                      </p>
                      <p className="mt-1 text-xs text-[#7A7868]">
                        {result.guardrail.stage} · {result.guardrail.reason_code}
                      </p>
                    </div>
                  )}

                  {result.status === "error" && (
                    <div className="border-l-2 border-[#C8A090] pl-4">
                      <p className="text-sm text-[#8B2E1E]">
                        {result.error ?? "Something went wrong."}
                      </p>
                    </div>
                  )}

                  {totalMs > 0 && (
                    <div className="rounded-lg border border-[#D0C8A8] bg-[#F5F1E3] px-4 py-3">
                      <p className="text-[10px] tracking-widest uppercase text-[#7A7868] font-semibold mb-2">
                        Timing breakdown
                      </p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs text-[#4A4A3A]">
                        {transcribeMs !== null && (
                          <div className="flex justify-between">
                            <span className="text-[#7A7868]">Transcribe</span>
                            <span className="font-medium tabular-nums">{transcribeMs.toFixed(1)} ms</span>
                          </div>
                        )}
                        {stage1Ms > 0 && (
                          <div className="flex justify-between">
                            <span className="text-[#7A7868]">Retrieve</span>
                            <span className="font-medium tabular-nums">{stage1Ms.toFixed(1)} ms</span>
                          </div>
                        )}
                        {generateMs !== null && (
                          <div className="flex justify-between">
                            <span className="text-[#7A7868]">Generate</span>
                            <span className="font-medium tabular-nums">{generateMs.toFixed(1)} ms</span>
                          </div>
                        )}
                        <div className="flex justify-between col-span-2 pt-1 border-t border-[#D0C8A8] mt-1">
                          <span className="text-[#4A4A3A] font-semibold">Total</span>
                          <span className="font-semibold tabular-nums text-[#1C3A20]">
                            {totalMs.toFixed(1)} ms
                          </span>
                        </div>
                      </div>
                      {comparisonMs.off !== null && comparisonMs.on !== null && (
                        <p className="mt-2 text-[11px] text-[#7A7868]">
                          Reranker off: {comparisonMs.off.toFixed(1)} ms · on:{" "}
                          {comparisonMs.on.toFixed(1)} ms
                        </p>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="lg:w-72 flex-shrink-0 flex flex-col gap-4">
            <div className="rounded-xl border border-[#D0C8A8] bg-[#F5F1E3] px-6 py-6">
              <p className="text-[10px] tracking-[0.2em] uppercase font-semibold text-[#7A7868] mb-4">
                Stage 1 · Clocked
              </p>

              <TimerDisplay ms={stage1Ms} budget={200} />

              <p className="mt-2 text-[11px] text-[#A8A090]">budget 200 ms</p>

              {result && stage1Ms > 0 ? (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mt-5 space-y-1.5"
                >
                  {[
                    { label: "Retrieve",   ms: stage1Ms   },
                    ...(transcribeMs !== null ? [{ label: "Transcribe", ms: transcribeMs }] : []),
                    ...(generateMs   !== null ? [{ label: "Generate",   ms: generateMs   }] : []),
                  ].map((row) => (
                    <div key={row.label} className="flex items-center gap-2 text-[11px]">
                      <span className="w-20 sm:w-[5.5rem] text-[#7A7868] flex-shrink-0">
                        {row.label}
                      </span>
                      <div className="flex-1 h-1 bg-[#D0C8A8] rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min((row.ms / (totalMs || 1)) * 100, 100)}%` }}
                          transition={{ duration: 0.5, ease: "easeOut" }}
                          className="h-full bg-[#1C3A20] rounded-full"
                        />
                      </div>
                      <span className="w-14 text-right tabular-nums text-[#4A4A3A] font-medium">
                        {row.ms.toFixed(1)} ms
                      </span>
                    </div>
                  ))}
                </motion.div>
              ) : (
                <p className="mt-6 text-[11px] text-[#A8A090] leading-relaxed">
                  Ask something to see the breakdown
                </p>
              )}
            </div>

            <div className="rounded-xl border border-[#D0C8A8] bg-[#F5F1E3] px-5 py-4">
              <p className="text-xs font-semibold text-[#1A1A0E] mb-1">How it works</p>
              <p className="text-[11px] text-[#7A7868] leading-relaxed">
                Stage 1 is the clocked path: transcript in, grounded answer out. Stage 2 (the
                full LLM response) sits outside the 200 ms budget — it is an order of magnitude
                slower and that is intentional.
              </p>
              <Link
                href="/benchmarks"
                className="mt-3 inline-block text-[11px] font-semibold text-[#1C3A20] hover:underline"
              >
                View full benchmarks →
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

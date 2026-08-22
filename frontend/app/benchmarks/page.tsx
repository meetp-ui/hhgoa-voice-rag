"use client";

import type React from "react";
import { motion, type Transition } from "framer-motion";
import Link from "next/link";

interface LatencyRow {
  config: string;
  p50: number; p70: number; p100: number;
  highlight?: boolean;
}

const LATENCY: LatencyRow[] = [
  { config: "Without rerank",                        p50: 152.5,   p70: 162.8,   p100: 309.9,   highlight: true },
  { config: "With rerank, top-3, int8 quantized",    p50: 684.3,   p70: 977.2,   p100: 1611.6   },
  { config: "With rerank, top-3 (fp32 ONNX)",        p50: 1046.1,  p70: 1185.6,  p100: 1804.7   },
  { config: "With rerank, top-5 (fp32 ONNX)",        p50: 1872.9,  p70: 2106.0,  p100: 2953.6   },
];

const LATENCY_META = {
  queries: 40,
  warmup: 3,
  chunks: 2144,
  rerankerFiredPct: 70,
  targetMs: 200,
};

const PER_STAGE: Array<{ stage: string; avgMs: number; note?: string }> = [
  { stage: "embed",          avgMs: 94.4  },
  { stage: "sparse_encode",  avgMs: 0.2   },
  { stage: "hybrid_search",  avgMs: 5.9   },
  { stage: "rerank",         avgMs: 1897.5, note: "queries where it fired" },
  { stage: "parent_expand",  avgMs: 44.7  },
];

interface ChunkRow {
  strategy: string;
  r5: number;
  highlight?: boolean;
}

const CHUNKING: ChunkRow[] = [
  { strategy: "structural",     r5: 0.842, highlight: true },
  { strategy: "fixed_overlap",  r5: 0.842, highlight: true },
  { strategy: "semantic",       r5: 0.789 },
];

const CHUNKING_META = {
  queryGroups: 38,
  passages: 380,
  split: "validation",
};

const RERANKER_NOTES = [
  {
    text: "Top-5 → top-3 candidates: P50 1872.9ms → 1046.1ms (-44%), P70 2106.0ms → 1185.6ms (-44%), P100 2953.6ms → 1804.7ms (-39%).",
    variant: "normal" as const,
  },
  {
    text: "int8 quantization: downloaded onnx-community/bge-reranker-v2-m3-ONNX's model_int8.onnx (570MB). Same ranking order as fp32, near-identical scores (0.9992 vs 0.9995).",
    variant: "normal" as const,
  },
  {
    text: "Combined (top-3 + int8): P50 684.3ms — a 63% reduction from the top-5 fp32 baseline. Still ~3.4x over the 200ms target.",
    variant: "highlight" as const,
  },
  {
    text: "Reranker ships opt-in, off by default. Without it, the 200ms target is met (P50=152.5ms, P70=162.8ms).",
    variant: "conclusion" as const,
  },
];

function fmt(n: number, decimals = 3) {
  return n.toFixed(decimals);
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] tracking-[0.2em] uppercase font-semibold text-[#7A7868] mb-4">
      {children}
    </p>
  );
}

function TableWrap({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-[#D0C8A8]">
      <table className="w-full text-sm">{children}</table>
    </div>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th
      className={`px-2 sm:px-4 py-2.5 sm:py-3 text-[9px] sm:text-[10px] tracking-[0.1em] sm:tracking-[0.15em] uppercase font-semibold text-[#7A7868] bg-[#F5F1E3] border-b border-[#D0C8A8] ${
        right ? "text-right" : "text-left"
      }`}
    >
      {children}
    </th>
  );
}

function Td({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return (
    <td
      className={`px-2 sm:px-4 py-2 sm:py-2.5 border-b border-[#E8E4D0] text-[11px] sm:text-[13px] ${
        right ? "text-right tabular-nums" : "text-left"
      } text-[#1A1A0E]`}
    >
      {children}
    </td>
  );
}


export default function Benchmarks() {
  const fadeIn = {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.35, ease: "easeOut" } as Transition,
  };

  return (
    <div className="min-h-screen bg-[#EEE9D1] text-[#1A1A0E]">
      <main className="mx-auto max-w-4xl px-4 sm:px-6 py-8 sm:py-10">

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-8 sm:mb-10">
          <motion.h1 {...fadeIn} className="text-2xl sm:text-3xl font-bold" style={{ fontFamily: 'var(--font-fraunces), Georgia, serif' }}>
            Benchmarks
          </motion.h1>
          <Link
            href="/"
            className="self-start rounded-full border border-[#D0C8A8] px-4 py-2 sm:py-1.5 text-sm text-[#4A4A3A] hover:border-[#1C3A20] hover:text-[#1C3A20] transition-colors whitespace-nowrap"
          >
            Back
          </Link>
        </div>

        <motion.section {...fadeIn} transition={{ delay: 0.05 }} className="mb-8 sm:mb-12">
          <SectionLabel>Retrieval Latency</SectionLabel>
          <p className="text-sm text-[#4A4A3A] leading-relaxed mb-4 sm:mb-6 max-w-2xl">
            Retrieval-only: embed query → hybrid search → optional rerank → parent-child expansion.
            Does not include STT or LLM generation. {LATENCY_META.queries} held-out queries
            ({LATENCY_META.warmup} warmup excluded) against the live {LATENCY_META.chunks.toLocaleString()}-chunk index.
            Target: {LATENCY_META.targetMs}ms for the retrieval-only path.
          </p>

          <TableWrap>
            <thead>
              <tr>
                <Th>Configuration</Th>
                <Th right>P50 ms</Th>
                <Th right>P70 ms</Th>
                <Th right>P100 ms</Th>
              </tr>
            </thead>
            <tbody>
              {LATENCY.map((row) => (
                <tr
                  key={row.config}
                  className={`transition-colors ${
                    row.highlight ? "bg-[#EDF5EE]" : "hover:bg-[#EEE9D1]"
                  }`}
                >
                  <Td>
                    <span className={row.highlight ? "font-semibold text-[#1C3A20]" : ""}>
                      {row.config}
                    </span>
                  </Td>
                  <Td right>{row.p50.toFixed(1)}</Td>
                  <Td right>{row.p70.toFixed(1)}</Td>
                  <Td right>{row.p100.toFixed(1)}</Td>
                </tr>
              ))}
            </tbody>
          </TableWrap>

          <p className="text-sm text-[#4A4A3A] mt-4 mb-5">
            <span className="font-medium">Reranker fired on {LATENCY_META.rerankerFiredPct}% of queries</span>
            {" · "}
            <span className="text-[#1C3A20] font-semibold">Without rerank, P50/P70 land under the {LATENCY_META.targetMs}ms target</span>
          </p>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mb-8 sm:mb-12"
        >
          <SectionLabel>Where the Time Goes</SectionLabel>
          <p className="text-sm text-[#4A4A3A] leading-relaxed mb-4 sm:mb-5 max-w-2xl">
            Top-5 fp32 run, average per stage. Reranker is ~20x the cost of every other stage combined.
          </p>
          <TableWrap>
            <thead>
              <tr>
                <Th>Stage</Th>
                <Th right>Avg ms</Th>
              </tr>
            </thead>
            <tbody>
              {PER_STAGE.map((row) => (
                <tr key={row.stage} className="hover:bg-[#EEE9D1] transition-colors">
                  <Td>
                    <code className="text-[12px] font-mono text-[#1A1A0E]">{row.stage}</code>
                    {row.note && (
                      <span className="text-[10px] text-[#7A7868] ml-2">({row.note})</span>
                    )}
                  </Td>
                  <Td right>{row.avgMs.toFixed(1)}</Td>
                </tr>
              ))}
            </tbody>
          </TableWrap>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8 sm:mb-12"
        >
          <SectionLabel>Chunking Ablation</SectionLabel>
          <p className="text-sm text-[#4A4A3A] leading-relaxed mb-4 sm:mb-5 max-w-2xl">
            {CHUNKING_META.queryGroups} query groups, {CHUNKING_META.passages} passages,{" "}
            {CHUNKING_META.split} split (MS MARCO-XI, Hindi). structural and fixed_overlap tie
            because MS MARCO-XI passages are short — only 0.5% exceed the 256-token threshold.
            semantic over-fragments already-coherent short passages, dropping recall.
          </p>
          <TableWrap>
            <thead>
              <tr>
                <Th>Strategy</Th>
                <Th right>Recall@5</Th>
              </tr>
            </thead>
            <tbody>
              {CHUNKING.map((row) => (
                <tr
                  key={row.strategy}
                  className={`transition-colors ${
                    row.highlight ? "bg-[#EDF5EE]" : "hover:bg-[#EEE9D1]"
                  }`}
                >
                  <Td>
                    <span className={row.highlight ? "font-semibold text-[#1C3A20]" : ""}>
                      {row.strategy}
                    </span>
                  </Td>
                  <Td right>{fmt(row.r5)}</Td>
                </tr>
              ))}
            </tbody>
          </TableWrap>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="mb-8 sm:mb-12"
        >
          <SectionLabel>Reranker Optimization</SectionLabel>
          <div className="flex flex-col gap-3">
            {RERANKER_NOTES.map((note, i) => (
              <div
                key={i}
                className={`rounded-xl border px-4 sm:px-5 py-3 sm:py-4 text-xs sm:text-sm leading-relaxed ${
                  note.variant === "highlight"
                    ? "border-[#B8D4BC] bg-[#EDF5EE] text-[#1C3A20] font-medium"
                    : note.variant === "conclusion"
                      ? "border-[#1C3A20] bg-[#1C3A20] text-[#EEE9D1] font-medium"
                      : "border-[#D0C8A8] bg-[#F5F1E3] text-[#4A4A3A]"
                }`}
              >
                {note.text}
              </div>
            ))}
          </div>
        </motion.section>

        <div className="flex justify-center pt-2 pb-6">
          <Link
            href="/"
            className="rounded-full border border-[#1C3A20] px-6 py-2.5 sm:py-2 text-sm font-medium text-[#1C3A20] hover:bg-[#1C3A20] hover:text-[#EEE9D1] transition-colors"
          >
            Back to Voice RAG
          </Link>
        </div>
      </main>
    </div>
  );
}

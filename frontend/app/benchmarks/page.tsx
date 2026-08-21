"use client";

import { motion, type Transition } from "framer-motion";
import Link from "next/link";

const LATENCY = {
  queries: 150,
  abstainedPct: 16.7,
  stage1: { p50: 33.6, p70: 37.6, p90: 42.4, p100: 54.8 },
  stage2: { p50: 721, p100: 1848 },
  notes: [
    {
      text: "Measured on hhgoa-voice-rag.fly.dev — the server answering your questions on the other tab — with no warm-up requests discarded. The same benchmark on an M4 MacBook Air gives p50 15.4 ms, p100 50.7 ms; shared vCPUs cost roughly 2× at the median, and the tail varies by ~20 ms run to run.",
      variant: "normal",
    },
    {
      text: "Caveat, stated rather than buried: these figures describe a machine that has already served traffic. Immediately after a deploy, first-touch page faults on the 332 MB memory-mapped index have produced outliers up to 461 ms — over budget. Warming one query per script at startup reduces this sharply but does not remove it. The machine never sleeps, so it is a deploy-time effect rather than something a visitor should hit.",
      variant: "caveat",
    },
    {
      text: "Stage 2 (complete LLM answer): p50 721 ms · p100 1848 ms — an order of magnitude slower, which is exactly why it sits outside the clock.",
      variant: "stage2",
    },
  ] as Array<{ text: string; variant: "normal" | "caveat" | "stage2" }>,
};

const PER_STAGE: Array<{ stage: string; p50: number; p100: number }> = [
  { stage: "mmr",         p50: 7.49, p100: 25.53 },
  { stage: "search",      p50: 4.34, p100: 33.91 },
  { stage: "extract",     p50: 2.12, p100: 14.78 },
  { stage: "fuse",        p50: 1.19, p100:  3.09 },
  { stage: "embed",       p50: 0.17, p100:  0.45 },
  { stage: "guard.input", p50: 0.01, p100:  0.21 },
  { stage: "guard.topic", p50: 0.00, p100:  0.03 },
];

interface ChunkRow {
  strategy: string;
  r5: number; r10: number; mrr10: number; ndcg10: number;
  p50ms: number; chunks: number;
  fused?: boolean;
}

const CHUNKING: ChunkRow[] = [
  { strategy: "fixed",           r5: 0.301, r10: 0.452, mrr10: 0.188, ndcg10: 0.247, p50ms:  5.89, chunks:   433_723 },
  { strategy: "sentence_window", r5: 0.364, r10: 0.464, mrr10: 0.225, ndcg10: 0.279, p50ms:  9.26, chunks:   825_152 },
  { strategy: "semantic",        r5: 0.344, r10: 0.459, mrr10: 0.217, ndcg10: 0.273, p50ms:  7.28, chunks:   643_840 },
  { strategy: "hierarchical",    r5: 0.371, r10: 0.484, mrr10: 0.248, ndcg10: 0.301, p50ms: 11.83, chunks: 1_038_408 },
  { strategy: "metadata_aware",  r5: 0.376, r10: 0.477, mrr10: 0.235, ndcg10: 0.290, p50ms:  4.15, chunks:   249_919 },
  {
    strategy: "FUSED(hierarchical+metadata_aware+sentence_window)",
    r5: 0.387, r10: 0.504, mrr10: 0.242, ndcg10: 0.302, p50ms: 16.60, chunks: 2_113_479,
    fused: true,
  },
];

interface LangRow {
  strategy: string;
  en: number; hi: number; ta: number; bn: number; mr: number;
  fused?: boolean;
}

const RECALL_BY_LANG: LangRow[] = [
  { strategy: "fixed",           en: 0.810, hi: 0.407, ta: 0.295, bn: 0.392, mr: 0.355 },
  { strategy: "sentence_window", en: 0.722, hi: 0.437, ta: 0.382, bn: 0.412, mr: 0.368 },
  { strategy: "semantic",        en: 0.735, hi: 0.437, ta: 0.363, bn: 0.382, mr: 0.380 },
  { strategy: "hierarchical",    en: 0.768, hi: 0.477, ta: 0.340, bn: 0.453, mr: 0.382 },
  { strategy: "metadata_aware",  en: 0.867, hi: 0.430, ta: 0.335, bn: 0.395, mr: 0.360 },
  {
    strategy: "FUSED(hierarchical+metadata_aware+sentence_window)",
    en: 0.808, hi: 0.477, ta: 0.385, bn: 0.473, mr: 0.377,
    fused: true,
  },
];

function fmt(n: number, decimals = 3) {
  return n.toFixed(decimals);
}

function fmtChunks(n: number) {
  return n.toLocaleString("en-US");
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

function Td({ children, right, fused }: { children: React.ReactNode; right?: boolean; fused?: boolean }) {
  return (
    <td
      className={`px-2 sm:px-4 py-2 sm:py-2.5 border-b border-[#E8E4D0] text-[11px] sm:text-[13px] ${
        right ? "text-right tabular-nums" : "text-left"
      } ${fused ? "italic text-[#4A4A3A]" : "text-[#1A1A0E]"}`}
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
          <SectionLabel>Latency</SectionLabel>
          <p className="text-sm text-[#4A4A3A] leading-relaxed mb-4 sm:mb-6 max-w-2xl">
            Stage 1 is the clocked path: transcript in, grounded answer out. Stage 2 is timed to
            the <em>complete</em> LLM answer — not time-to-first-token, which would misrepresent
            it. Speech-to-text is measured separately and is not in either figure.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
            {(["p50", "p70", "p90", "p100"] as const).map((pct, i) => {
              const val =
                pct === "p50" ? LATENCY.stage1.p50
                : pct === "p70" ? LATENCY.stage1.p70
                : pct === "p90" ? LATENCY.stage1.p90
                : LATENCY.stage1.p100;
              return (
                <motion.div
                  key={pct}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + i * 0.06 }}
                  className="rounded-xl border border-[#D0C8A8] bg-[#F5F1E3] px-5 py-4"
                >
                  <p className="text-[10px] tracking-[0.15em] uppercase text-[#7A7868] font-semibold mb-1">
                    {pct.toUpperCase()}
                  </p>
                  <div className="flex items-baseline gap-1">
                    <span className="text-2xl sm:text-3xl font-bold text-[#1C3A20]" style={{ fontFamily: 'var(--font-fraunces), Georgia, serif' }}>
                      {val}
                    </span>
                    <span className="text-xs sm:text-sm text-[#4A7A52]">ms</span>
                  </div>
                </motion.div>
              );
            })}
          </div>

          <p className="text-sm text-[#4A4A3A] mb-5">
            <span className="font-medium">{LATENCY.queries} queries</span>
            {" · "}
            <span className="text-[#1C3A20] font-semibold">p100 under the 200 ms target</span>
            {" · "}
            abstained on {LATENCY.abstainedPct}%
          </p>

          <div className="flex flex-col gap-3">
            {LATENCY.notes.map((note, i) => (
              <div
                key={i}
                className={`rounded-xl border px-4 sm:px-5 py-3 sm:py-4 text-xs sm:text-sm leading-relaxed ${
                  note.variant === "caveat"
                    ? "border-[#D0C8A8] bg-[#F5F1E3] text-[#4A4A3A]"
                    : note.variant === "stage2"
                      ? "border-[#B8D4BC] bg-[#EDF5EE] text-[#1C3A20] font-medium"
                      : "border-[#D0C8A8] bg-[#F5F1E3] text-[#4A4A3A]"
                }`}
              >
                {note.variant === "caveat" && (
                  <strong className="text-[#1A1A0E]">Caveat, stated rather than buried: </strong>
                )}
                {note.text}
              </div>
            ))}
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mb-8 sm:mb-12"
        >
          <SectionLabel>Per Stage</SectionLabel>
          <TableWrap>
            <thead>
              <tr>
                <Th>Stage</Th>
                <Th right>P50 ms</Th>
                <Th right>P100 ms</Th>
              </tr>
            </thead>
            <tbody>
              {PER_STAGE.map((row) => (
                <tr key={row.stage} className="hover:bg-[#EEE9D1] transition-colors">
                  <Td>
                    <code className="text-[12px] font-mono text-[#1A1A0E]">{row.stage}</code>
                  </Td>
                  <Td right>{row.p50.toFixed(2)}</Td>
                  <Td right>{row.p100.toFixed(2)}</Td>
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
            Five strategies scored against MSMARCO-XI&apos;s own{" "}
            <code className="text-[12px] bg-[#E4DFCA] px-1.5 py-0.5 rounded">is_selected</code>{" "}
            ground truth on 500 held-out queries. The fused row tests whether querying several
            strategies at once beats the best single one — it did not, at the k we actually use,
            so a single strategy ships.
          </p>
          <TableWrap>
            <thead>
              <tr>
                <Th>Strategy</Th>
                <Th right>R@5</Th>
                <Th right>R@10</Th>
                <Th right>MRR@10</Th>
                <Th right>NDCG@10</Th>
                <Th right>P50 ms</Th>
                <Th right>Chunks</Th>
              </tr>
            </thead>
            <tbody>
              {CHUNKING.map((row) => (
                <tr
                  key={row.strategy}
                  className={`transition-colors ${
                    row.fused ? "bg-[#F0EDE0]" : "hover:bg-[#EEE9D1]"
                  }`}
                >
                  <Td fused={row.fused}>
                    {row.fused ? <em>{row.strategy}</em> : row.strategy}
                  </Td>
                  <Td right fused={row.fused}>{fmt(row.r5)}</Td>
                  <Td right fused={row.fused}>{fmt(row.r10)}</Td>
                  <Td right fused={row.fused}>{fmt(row.mrr10)}</Td>
                  <Td right fused={row.fused}>{fmt(row.ndcg10)}</Td>
                  <Td right fused={row.fused}>{row.p50ms.toFixed(2)}</Td>
                  <Td right fused={row.fused}>{fmtChunks(row.chunks)}</Td>
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
          <SectionLabel>Recall@10 by Language</SectionLabel>
          <p className="text-sm text-[#4A4A3A] leading-relaxed mb-4 sm:mb-5 max-w-2xl">
            English retrieves markedly better than the Indic languages. Static embeddings degrade
            on morphologically rich scripts, which is why sparse retrieval is routed by script
            rather than shared.
          </p>
          <TableWrap>
            <thead>
              <tr>
                <Th>Strategy</Th>
                <Th right>EN</Th>
                <Th right>HI</Th>
                <Th right>TA</Th>
                <Th right>BN</Th>
                <Th right>MR</Th>
              </tr>
            </thead>
            <tbody>
              {RECALL_BY_LANG.map((row) => (
                <tr
                  key={row.strategy}
                  className={`transition-colors ${
                    row.fused ? "bg-[#F0EDE0]" : "hover:bg-[#EEE9D1]"
                  }`}
                >
                  <Td fused={row.fused}>
                    {row.fused ? <em>{row.strategy}</em> : row.strategy}
                  </Td>
                  <Td right fused={row.fused}>{fmt(row.en)}</Td>
                  <Td right fused={row.fused}>{fmt(row.hi)}</Td>
                  <Td right fused={row.fused}>{fmt(row.ta)}</Td>
                  <Td right fused={row.fused}>{fmt(row.bn)}</Td>
                  <Td right fused={row.fused}>{fmt(row.mr)}</Td>
                </tr>
              ))}
            </tbody>
          </TableWrap>
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

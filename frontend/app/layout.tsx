import type { Metadata } from "next";
import { Fraunces, Noto_Sans, Noto_Sans_Devanagari } from "next/font/google";
import type React from "react";
import "./globals.css";

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  axes: ["opsz", "SOFT", "WONK"],
});

const notoSans = Noto_Sans({
  variable: "--font-noto-sans",
  subsets: ["latin"],
});

const notoSansDevanagari = Noto_Sans_Devanagari({
  variable: "--font-noto-devanagari",
  subsets: ["devanagari", "latin"],
});

export const metadata: Metadata = {
  title: "Voice RAG over MSMARCO-XI",
  description:
    "Ask a question in any of five languages and search 250,000 passages in under 200 ms.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${notoSans.variable} ${notoSansDevanagari.variable}`}
      style={{ height: "100%", maxWidth: "100vw", overflowX: "hidden" }}
    >
      <body style={{
        minHeight: "100%",
        display: "flex",
        flexDirection: "column",
        color: "var(--bark)",
        background: "var(--cream)",
        fontFamily: "var(--font-noto-sans), Arial, Helvetica, sans-serif",
        WebkitFontSmoothing: "antialiased",
        MozOsxFontSmoothing: "grayscale",
        margin: 0,
        padding: 0,
        boxSizing: "border-box",
        overflowX: "hidden",
      } as React.CSSProperties}>
        {children}
      </body>
    </html>
  );
}

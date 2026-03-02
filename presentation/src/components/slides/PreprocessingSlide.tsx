"use client";

import { motion } from "framer-motion";
import { Filter, ArrowDown } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.1 } } };
const item = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } };

const steps = [
  { label: "Raw Collection", count: "17,552", detail: "~50h audio from 3 sources", color: "bg-blue-500" },
  { label: "Audio Validation", count: "–312", detail: "Corrupted/unreadable files removed", color: "bg-blue-400" },
  { label: "Duration Filter", count: "–1,847", detail: "< 0.5s or > 30s removed", color: "bg-purple-500" },
  { label: "Text Validation", count: "–2,215", detail: "Non-Myanmar characters, empty transcripts removed", color: "bg-violet-500" },
  { label: "SNR Filter", count: "–528", detail: "Noisy samples below threshold removed", color: "bg-amber-500" },
  { label: "Deduplication", count: "–351", detail: "Exact & near-duplicate transcripts removed", color: "bg-orange-500" },
  { label: "Unicode Normalization", count: "—", detail: "Standardized Myanmar Unicode (NFC form)", color: "bg-pink-500" },
  { label: "Clean Data", count: "12,190", detail: "30% removed as low-quality", color: "bg-emerald-500" },
  { label: "Speed Augmentation", count: "+10,515", detail: "0.9x + 1.1x speed (preserves pitch, train only)", color: "bg-teal-500" },
  { label: "Final Dataset", count: "22,705", detail: "54.2 hours total", color: "bg-green-500" },
];

export default function PreprocessingSlide({ onImageClick }: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title">Data Preprocessing &amp; Cleaning</h2>

      <div className="grid md:grid-cols-5 gap-8 flex-1">
        {/* left: pipeline steps */}
        <motion.div
          variants={stagger}
          initial="hidden"
          animate="show"
          className="md:col-span-3 flex flex-col gap-4"
        >
          <h3 className="text-2xl md:text-3xl uppercase tracking-wider text-accent font-extrabold mb-2 flex items-center gap-2">
            <Filter size={20} /> Pipeline Steps
          </h3>
          <div className="flex flex-col gap-2 flex-1">
            {steps.map((s, i) => (
              <motion.div key={s.label} variants={item}>
                <div className="flex items-center gap-3">
                  {/* color dot & line */}
                  <div className="flex flex-col items-center">
                    <div className={`w-3 h-3 rounded-full ${s.color}`} />
                    {i < steps.length - 1 && (
                      <div className="w-px h-4 bg-slate-200" />
                    )}
                  </div>
                  {/* content */}
                  <div className="flex-1 flex items-baseline gap-3 py-1.5">
                    <span className="font-bold text-2xl md:text-xl min-w-[170px] text-slate-800">{s.label}</span>
                    <span
                      className={`font-mono text-2xl font-bold tabular-nums ${
                        s.count.startsWith("–")
                          ? "text-red-500"
                          : s.count.startsWith("+")
                          ? "text-emerald-600"
                          : s.count === "—"
                          ? "text-slate-700"
                          : "text-accent"
                      }`}
                    >
                      {s.count}
                    </span>
                    <span className="text-xl text-slate-700">{s.detail}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-3 glass text-2xl text-slate-600 flex items-center gap-2">
            <ArrowDown size={14} className="text-emerald-600" />
            <span>
              Reduced <span className="text-slate-900 font-extrabold">30%</span> of raw data &mdash; improved training stability &amp; consistency.
              Augmentation increases diversity without new data collection.
            </span>
          </div>
        </motion.div>

        {/* right: charts */}
        <div className="md:col-span-2 flex flex-col gap-4">
          <div className="glass flex-1 flex flex-col">
            <h3 className="text-2xl uppercase tracking-wider text-slate-700 font-extrabold mb-3">
              Data Processing Funnel
            </h3>
            <img
              src="/charts/12_data_processing_funnel.png"
              alt="Data Processing Funnel"
              className="chart-thumb rounded-lg flex-1 object-contain"
              onClick={() =>
                onImageClick("/charts/12_data_processing_funnel.png", "Data Processing Funnel")
              }
            />
          </div>
          <div className="glass flex-1 flex flex-col">
            <h3 className="text-2xl uppercase tracking-wider text-slate-700 font-extrabold mb-3">
              Duration &amp; Text Distribution
            </h3>
            <img
              src="/charts/04_duration_vs_text.png"
              alt="Duration vs Text Length"
              className="chart-thumb rounded-lg flex-1 object-contain"
              onClick={() =>
                onImageClick("/charts/04_duration_vs_text.png", "Duration vs Text Length Distribution")
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}

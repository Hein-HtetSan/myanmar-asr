"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Trophy, TrendingDown, BarChart3, ChevronRight, Lightbulb } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.08 } } };
const item = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } };

const testResults = [
  { model: "Whisper v3 Turbo", wer: "54.49*", cer: "36.00*", loss: "1.470", time: "159 min", werNum: 54.49, cerNum: 36.0, mColor: "text-red-600", bg: "bg-red-50", border: "border-red-200" },
  { model: "Dolphin", wer: "33.02", cer: "28.00", loss: "1.451", time: "335 min", werNum: 33.02, cerNum: 28.0, mColor: "text-blue-600", bg: "bg-blue-50", border: "border-blue-200" },
  { model: "SeamlessM4T v2", wer: "49.12", cer: "13.04", loss: "2.070", time: "239 min", werNum: 49.12, cerNum: 13.04, mColor: "text-emerald-600", bg: "bg-emerald-50", border: "border-emerald-200" },
];

const valResults = [
  { model: "Whisper v3 Turbo", wer: "54.49 (step 4368)", cer: "36.00 (step 4368)", loss: "1.470" },
  { model: "Dolphin", wer: "34.95 (step 5346)", cer: "29.49 (step 6318)", loss: "1.430" },
  { model: "SeamlessM4T v2", wer: "47.99 (step 3276)", cer: "12.56 (step 3822)", loss: "2.029" },
];

const keyFindings = [
  {
    title: "Dolphin: Best word-level accuracy (WER = 33.02%)",
    points: [
      "Whisper-large-v2 architecture with 32 decoder layers benefits from deeper language modeling",
      "Cosine LR + label smoothing + weight decay provides strong regularization",
      "Longest training (335 min / 15 epochs) but best WER",
    ],
  },
  {
    title: "SeamlessM4T: Best character-level accuracy (CER = 13.04%)",
    points: [
      "w2v-BERT 2.0 encoder captures fine-grained acoustic features",
      "Length adaptor compresses speech representations effectively",
      "Character-level precision is 2.1x better than Dolphin despite higher WER",
    ],
  },
  {
    title: "WER vs CER divergence — expected for Myanmar",
    points: [
      "Myanmar script has no spaces between words ➞ word boundary errors inflate WER",
      "CER is more meaningful for Myanmar because the script is character-based",
      "SeamlessM4T gets characters right but struggles with word segmentation",
    ],
  },
];

const trainingNotes = [
  { model: "Whisper Turbo", note: "Fast convergence in 5 epochs, but plateaus early — limited by shallow 4-layer decoder", color: "text-red-600" },
  { model: "Dolphin", note: "Steady improvement over 15 epochs, strongest at word-level predictions", color: "text-blue-600" },
  { model: "SeamlessM4T", note: "Rapid CER reduction; converges by epoch 9, outstanding character accuracy", color: "text-emerald-600" },
];

const charts = [
  { src: "/charts/06_model_comparison_bar.png", label: "WER / CER Comparison" },
  { src: "/charts/14_test_results_bar.png", label: "Test Results" },
  { src: "/charts/07_training_curves_wer_cer.png", label: "Training Curves" },
  { src: "/charts/08_train_loss_curves.png", label: "Loss Curves" },
  { src: "/charts/13_eval_loss_curves.png", label: "Eval Loss" },
  { src: "/charts/09_model_summary_table.png", label: "Summary Table" },
  { src: "/charts/10_radar_model_strengths.png", label: "Radar Comparison" },
];

export default function ResultsSlide({ onImageClick }: SlideProps) {
  const [chartIdx, setChartIdx] = useState(0);
  const [tab, setTab] = useState<"test" | "val" | "findings">("test");

  return (
    <div className="slide">
      <h2 className="slide-title">Performance Evaluation</h2>

      <div className="grid md:grid-cols-5 gap-8 flex-1">
        {/* left: metrics */}
        <div className="md:col-span-3 flex flex-col gap-6">
          {/* sub-tabs */}
          <div className="flex flex-wrap gap-2 text-xl">
            {[
              { id: "test" as const, label: "Test Results" },
              { id: "val" as const, label: "Best Validation" },
              { id: "findings" as const, label: "Key Findings" },
            ].map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-2 border-2 border-slate-900 font-extrabold transition-all ${
                  tab === t.id
                    ? "bg-emerald-400 text-slate-900 shadow-[2px_2px_0_0_rgba(15,23,42,1)] translate-x-[-2px] translate-y-[-2px]"
                    : "bg-white text-slate-700 hover:bg-slate-100"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {tab === "test" && (
              <motion.div key="test" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-6">
                {/* metric cards */}
                <motion.div variants={stagger} initial="hidden" animate="show" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {testResults.map((r) => (
                    <motion.div key={r.model} variants={item} className={`glass ${r.bg} ${r.border} flex flex-col`}>
                      <div className={`text-lg md:text-xl font-extrabold ${r.mColor} uppercase tracking-wider mb-3 leading-tight`}>
                        {r.model}
                      </div>
                      <div className="flex flex-wrap gap-4 md:gap-6 mb-3">
                        <div>
                          <div className="text-lg md:text-xl text-slate-700 font-bold mb-1">WER</div>
                          <div className={`text-2xl font-extrabold tabular-nums ${
                            r.werNum <= 33.02 ? "text-blue-600" : "text-slate-700"
                          }`}>
                            {r.wer}%
                          </div>
                        </div>
                        <div>
                          <div className="text-lg md:text-xl text-slate-700 font-bold mb-1">CER</div>
                          <div className={`text-2xl font-extrabold tabular-nums ${
                            r.cerNum <= 13.04 ? "text-emerald-600" : "text-slate-700"
                          }`}>
                            {r.cer}%
                          </div>
                        </div>
                      </div>
                      <div className="text-lg md:text-xl text-slate-700 mt-auto flex flex-col gap-1 border-t border-slate-200/50 pt-3">
                        <span className="font-mono bg-white/50 px-2 py-0.5 rounded w-max shadow-sm border border-slate-200/50">L {r.loss}</span>
                        <span className="font-mono bg-white/50 px-2 py-0.5 rounded w-max shadow-sm border border-slate-200/50">T {r.time}</span>
                      </div>
                    </motion.div>
                  ))}
                </motion.div>

                {/* winner badges */}
                <div className="flex gap-3 flex-wrap">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-2xl">
                    <Trophy size={14} className="text-blue-600" />
                    <span className="text-blue-700 font-extrabold">Best WER:</span>
                    <span className="text-slate-800">Dolphin — 33.02%</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-2xl">
                    <Trophy size={14} className="text-emerald-600" />
                    <span className="text-emerald-700 font-extrabold">Best CER:</span>
                    <span className="text-slate-800">SeamlessM4T — 13.04%</span>
                  </div>
                </div>

                {/* improvement */}
                <div className="glass">
                  <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-4 flex items-center gap-2">
                    <TrendingDown size={18} /> Improvement from Zero-Shot
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xl lg:text-2xl">
                    <div>
                      <span className="text-slate-700">WER: </span>
                      <span className="text-red-500">~100%</span>
                      <span className="text-slate-700"> ➞ </span>
                      <span className="text-blue-600 font-bold">33.02%</span>
                      <span className="text-emerald-600 ml-2 font-mono text-xl">⬇ 67pp</span>
                    </div>
                    <div>
                      <span className="text-slate-700">CER: </span>
                      <span className="text-red-500">~88%</span>
                      <span className="text-slate-700"> ➞ </span>
                      <span className="text-emerald-600 font-bold">13.04%</span>
                      <span className="text-emerald-600 ml-2 font-mono text-xl">⬇ 75pp</span>
                    </div>
                  </div>
                </div>

                <p className="text-xl text-slate-700">
                  * Whisper Turbo: best validation scores (no separate test run logged)
                </p>
              </motion.div>
            )}

            {tab === "val" && (
              <motion.div key="val" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-5">
                <div className="glass overflow-x-auto">
                  <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-4">
                    Best Validation Scores
                  </h3>
                  <table className="data-table text-xl md:text-2xl w-full">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Best Val WER</th>
                        <th>Best Val CER</th>
                        <th>Best Eval Loss</th>
                      </tr>
                    </thead>
                    <tbody>
                      {valResults.map((r) => (
                        <tr key={r.model}>
                          <td className="font-bold text-slate-800">{r.model}</td>
                          <td className="font-mono text-slate-600">{r.wer}</td>
                          <td className="font-mono text-slate-600">{r.cer}</td>
                          <td className="font-mono text-slate-600">{r.loss}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="glass">
                  <h3 className="text-2xl uppercase tracking-wider text-slate-700 font-extrabold mb-3">
                    Training Curves Summary
                  </h3>
                  <div className="space-y-2">
                    {trainingNotes.map((t) => (
                      <div key={t.model} className="flex flex-col lg:flex-row lg:items-start lg:gap-3 text-xl md:text-2xl pb-3">
                        <span className={`font-extrabold min-w-[150px] lg:min-w-[170px] flex-shrink-0 ${t.color}`}>{t.model}:</span>
                        <span className="text-slate-600">{t.note}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {tab === "findings" && (
              <motion.div key="findings" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-5">
                {keyFindings.map((f, i) => (
                  <div key={i} className="glass">
                    <h4 className="text-2xl font-extrabold text-slate-800 mb-2 flex items-center gap-2">
                      <Lightbulb size={14} className="text-amber-500" />
                      {f.title}
                    </h4>
                    <ul className="space-y-1 text-2xl text-slate-600">
                      {f.points.map((p, j) => (
                        <li key={j} className="flex items-start gap-2">
                          <span className="text-slate-700 mt-1">•</span>
                          <span>{p}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* right: chart carousel */}
        <div className="md:col-span-2 flex flex-col gap-3">
          <div className="glass flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-2xl uppercase tracking-wider text-slate-700 font-extrabold flex items-center gap-2">
                <BarChart3 size={15} /> Charts
              </h3>
              <div className="flex gap-1">
                {charts.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setChartIdx(i)}
                    className={`w-2 h-2 rounded-full transition-all ${
                      i === chartIdx ? "bg-accent w-5" : "bg-slate-200 hover:bg-slate-300"
                    }`}
                  />
                ))}
              </div>
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={chartIdx}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="flex-1 flex flex-col"
              >
                <img
                  src={charts[chartIdx].src}
                  alt={charts[chartIdx].label}
                  className="chart-thumb rounded-lg flex-1 object-contain"
                  onClick={() => onImageClick(charts[chartIdx].src, charts[chartIdx].label)}
                />
                <p className="text-xl text-slate-700 text-center mt-2">
                  {charts[chartIdx].label}
                  <span className="text-slate-700 ml-1">— click to enlarge</span>
                </p>
              </motion.div>
            </AnimatePresence>

            <button
              onClick={() => setChartIdx((i) => (i + 1) % charts.length)}
              className="mt-2 text-xl text-accent flex items-center justify-center gap-1 hover:underline"
            >
              Next chart <ChevronRight size={12} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Lightbulb, Rocket, ArrowRight } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.08 } } };
const item = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } };

const achievements = [
  "Curated 54.2-hour Myanmar ASR dataset (22,705 samples from 3 sources + augmentation)",
  "Deep cleaning pipeline — removed 30% low-quality data for improved training",
  "Successfully fine-tuned all 3 multilingual speech models on Myanmar",
  "CER reduced from ~88% to 13.04% (SeamlessM4T) — exceeding < 35% target by wide margin",
  "WER reduced from ~100% to 33.02% (Dolphin / Whisper-large-v2)",
  "Systematic experiment tracking with MLflow for full reproducibility",
];

const takeaways = [
  { text: "Frozen encoder is critical for low-resource ASR — prevents overfitting while preserving multilingual features", color: "text-amber-700" },
  { text: "CER is more appropriate than WER for agglutinative languages like Myanmar", color: "text-blue-700" },
  { text: "Deeper decoders (32 layers) give better WER; adaptor architectures excel at CER", color: "text-emerald-700" },
  { text: "Cosine LR + label smoothing consistently outperform simple linear decay", color: "text-purple-700" },
  { text: "54 hours of clean data with proper augmentation produces competitive results via fine-tuning", color: "text-red-700" },
];

const phases = [
  { phase: "Fine-tune Whisper v3 Turbo", status: "done" },
  { phase: "Fine-tune Dolphin (Whisper-large-v2)", status: "done" },
  { phase: "Fine-tune SeamlessM4T v2 Large", status: "done" },
  { phase: "Model optimization (quantization, CTranslate2)", status: "next" },
  { phase: "Web demo deployment (FastAPI + WebSocket)", status: "planned" },
  { phase: "Expand dataset (VOA Myanmar, Common Voice)", status: "planned" },
];

export default function ConclusionSlide({ onImageClick }: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title">Conclusion &amp; Future Plan</h2>

      <div className="grid md:grid-cols-3 gap-8 flex-1">
        {/* left: achievements */}
        <div className="flex flex-col gap-6">
          <h3 className="text-2xl md:text-3xl uppercase tracking-wider text-emerald-600 font-extrabold flex items-center gap-2">
            <CheckCircle2 size={24} /> Key Achievements
          </h3>
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="show"
            className="space-y-4 flex-1"
          >
            {achievements.map((a, i) => (
              <motion.div
                key={i}
                variants={item}
                className="flex items-start gap-3 text-xl md:text-2xl text-slate-800 font-medium"
              >
                <CheckCircle2 size={24} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                <span className="leading-snug">{a}</span>
              </motion.div>
            ))}
          </motion.div>

          {/* summary card */}
          <img
            src="/charts/15_final_summary_scorecard.png"
            alt="Final Summary Scorecard"
            className="chart-thumb rounded-lg"
            onClick={() =>
              onImageClick("/charts/15_final_summary_scorecard.png", "Final Summary Scorecard")
            }
          />
        </div>

        {/* middle: takeaways */}
        <div className="flex flex-col gap-6">
          <h3 className="text-2xl md:text-3xl uppercase tracking-wider text-amber-600 font-extrabold flex items-center gap-2">
            <Lightbulb size={24} /> Key Takeaways
          </h3>
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="show"
            className="space-y-4 flex-1"
          >
            {takeaways.map((t, i) => (
              <motion.div key={i} variants={item} className="glass !p-4 border-l-4 border-l-amber-500">
                <p className={`text-xl md:text-2xl font-bold text-slate-800 leading-snug`}>{t.text}</p>
              </motion.div>
            ))}
          </motion.div>

          {/* comparative winner */}
          <div className="glass text-center !p-4">
            <div className="text-xl md:text-2xl text-slate-900 font-extrabold uppercase tracking-wider mb-4 border-b-2 border-slate-900 pb-2 inline-block">Category Winners</div>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-blue-50 p-2 rounded border-2 border-slate-900 shadow-[2px_2px_0_0_rgba(15,23,42,1)]">
                <div className="text-sm md:text-lg font-bold text-slate-700 uppercase tracking-widest mb-1">Best WER</div>
                <div className="text-xl md:text-2xl font-black text-blue-700">Dolphin</div>
                <div className="text-lg font-mono text-blue-900 mt-1">33.02%</div>
              </div>
              <div className="bg-emerald-50 p-2 rounded border-2 border-slate-900 shadow-[2px_2px_0_0_rgba(15,23,42,1)]">
                <div className="text-sm md:text-lg font-bold text-slate-700 uppercase tracking-widest mb-1">Best CER</div>
                <div className="text-xl md:text-2xl font-black text-emerald-700">SeamlessM4T</div>
                <div className="text-lg font-mono text-emerald-900 mt-1">13.04%</div>
              </div>
              <div className="bg-red-50 p-2 rounded border-2 border-slate-900 shadow-[2px_2px_0_0_rgba(15,23,42,1)]">
                <div className="text-sm md:text-lg font-bold text-slate-700 uppercase tracking-widest mb-1">Fastest</div>
                <div className="text-xl md:text-2xl font-black text-red-700">Whisper Turbo</div>
                <div className="text-lg font-mono text-red-900 mt-1">159 min</div>
              </div>
              <div className="bg-purple-50 p-2 rounded border-2 border-slate-900 shadow-[2px_2px_0_0_rgba(15,23,42,1)]">
                <div className="text-sm md:text-lg font-bold text-slate-700 uppercase tracking-widest mb-1">Reg.</div>
                <div className="text-xl md:text-2xl font-black text-purple-700">Dolphin</div>
                <div className="text-lg font-mono text-purple-900 mt-1">Val loss 1.43</div>
              </div>
            </div>
          </div>
        </div>

        {/* right: future plan */}
        <div className="flex flex-col gap-6">
          <h3 className="text-2xl md:text-3xl uppercase tracking-wider text-accent font-extrabold flex items-center gap-2">
            <Rocket size={24} /> Roadmap
          </h3>
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="show"
            className="space-y-4 flex-1"
          >
            {phases.map((p, i) => (
              <motion.div
                key={i}
                variants={item}
                className="flex items-center gap-4 p-2 bg-slate-50 border border-slate-200 rounded"
              >
                <div className="flex flex-col items-center">
                  <div
                    className={`w-4 h-4 rounded-full border-2 border-slate-900 ${
                      p.status === "done"
                        ? "bg-emerald-500"
                        : p.status === "next"
                        ? "bg-amber-400"
                        : "bg-white"
                    }`}
                  />
                  {i < phases.length - 1 && <div className="w-0.5 h-8 bg-slate-900 absolute mt-5 -z-10" />}
                </div>
                <div className="flex-1 flex flex-col items-start justify-center gap-1 py-1">
                  <span className="text-xl md:text-2xl font-bold text-slate-800 leading-tight">{p.phase}</span>
                  <span
                    className={`badge text-sm md:text-base font-bold uppercase tracking-wider ${
                      p.status === "done"
                        ? "bg-emerald-200 text-emerald-900 border-2 border-slate-900 shadow-[2px_2px_0_0_rgba(15,23,42,1)]"
                        : p.status === "next"
                        ? "bg-amber-200 text-amber-900 border-2 border-slate-900 shadow-[2px_2px_0_0_rgba(15,23,42,1)]"
                        : "bg-white text-slate-700 border-2 border-slate-900 border-dashed"
                    }`}
                  >
                    {p.status === "done" ? <span><CheckCircle2 size={14} className="inline mr-1 stroke-[3px]" /> Done</span> : p.status === "next" ? <span><ArrowRight size={14} className="inline mr-1 stroke-[3px]" /> Next</span> : "Planned"}
                  </span>
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* deliverables */}
          <div className="glass !p-4 bg-slate-900 text-white shadow-[6px_6px_0_0_rgba(52,211,153,1)]">
            <div className="text-xl md:text-2xl text-emerald-400 font-extrabold uppercase tracking-wider mb-3 border-b-2 border-slate-700 pb-2">Deliverables</div>
            <ul className="text-xl text-slate-200 space-y-2 font-bold">
              <li className="flex items-start gap-2"><ArrowRight size={20} className="text-emerald-400 mt-1 shrink-0" /> 3 fine-tuned Myanmar ASR models</li>
              <li className="flex items-start gap-2"><ArrowRight size={20} className="text-emerald-400 mt-1 shrink-0" /> Comparative study &amp; tech report</li>
              <li className="flex items-start gap-2"><ArrowRight size={20} className="text-emerald-400 mt-1 shrink-0" /> Open dataset &amp; models on HuggingFace</li>
              <li className="flex items-start gap-2"><ArrowRight size={20} className="text-emerald-400 mt-1 shrink-0" /> Web demo for real-time inference</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

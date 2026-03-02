"use client";

import { motion } from "framer-motion";
import { Server, Zap, MonitorSmartphone, ExternalLink, Play } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.1 } } };
const item = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } };

const selectionCards = [
  {
    task: "Word-level tasks",
    examples: "Search, indexing, NLP",
    model: "Dolphin",
    metric: "WER 33.02%",
    color: "border-slate-900 bg-blue-50",
    accent: "text-blue-700",
  },
  {
    task: "Character-level tasks",
    examples: "Subtitles, display, OCR",
    model: "SeamlessM4T",
    metric: "CER 13.04%",
    color: "border-slate-900 bg-emerald-50",
    accent: "text-emerald-700",
  },
  {
    task: "Low-latency tasks",
    examples: "Real-time, streaming",
    model: "Whisper Turbo",
    metric: "Fastest inference",
    color: "border-slate-900 bg-red-50",
    accent: "text-red-700",
  },
];

const optimizations = [
  { name: "CTranslate2", desc: "2-4x faster inference than PyTorch" },
  { name: "Quantization", desc: "INT8/FP16 for CPU deployment" },
  { name: "Beam search tuning", desc: "Optimize speed/accuracy trade-off" },
  { name: "Batch inference", desc: "Process multiple audio files simultaneously" },
];

export default function DeploymentSlide(_props: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title">System Deployment</h2>
      <p className="slide-subtitle">Planned deployment architecture and model selection strategy</p>

      <div className="grid md:grid-cols-2 gap-5 flex-1">
        {/* left: architecture + model selection */}
        <div className="flex flex-col gap-5">
          {/* architecture diagram */}
          <div className="glass">
            <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-4 flex items-center gap-2">
              <Server size={15} /> Deployment Architecture
            </h3>
            <div className="flex items-center gap-3 justify-center py-4">
              {[
                { label: "Client", sub: "Browser/Mobile", icon: <MonitorSmartphone size={20} />, bg: "bg-purple-50 border-slate-900" },
                { label: "API Server", sub: "FastAPI + WebSocket", icon: <Server size={20} />, bg: "bg-blue-50 border-slate-900" },
                { label: "ASR Model", sub: "Best per task", icon: <Zap size={20} />, bg: "bg-emerald-50 border-slate-900" },
              ].map((b, i) => (
                <div key={b.label} className="flex items-center gap-3">
                  <div className={`rounded-xl border p-4 text-center min-w-[120px] ${b.bg}`}>
                    <div className="flex justify-center mb-2 text-slate-700">{b.icon}</div>
                    <div className="text-2xl font-extrabold text-slate-800">{b.label}</div>
                    <div className="text-xl text-slate-700">{b.sub}</div>
                  </div>
                  {i < 2 && <span className="text-slate-700 text-2xl">➞</span>}
                </div>
              ))}
            </div>
          </div>

          {/* model selection */}
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="show"
            className="flex flex-col gap-2"
          >
            <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-1">
              Model Selection Strategy
            </h3>
            {selectionCards.map((c) => (
              <motion.div key={c.task} variants={item} className={`glass border ${c.color} flex items-center gap-4`}>
                <div className="flex-1">
                  <div className="text-2xl font-extrabold text-slate-800">{c.task}</div>
                  <div className="text-xl text-slate-700">{c.examples}</div>
                </div>
                <div className="text-right">
                  <div className={`text-2xl font-bold ${c.accent}`}>{c.model}</div>
                  <div className="text-xl text-slate-700 font-mono">{c.metric}</div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* right: optimizations + demo */}
        <div className="flex flex-col gap-5">
          <div className="glass flex-1">
            <h3 className="text-2xl uppercase tracking-wider text-amber-600 font-extrabold mb-4 flex items-center gap-2">
              <Zap size={15} /> Inference Optimization (Planned)
            </h3>
            <div className="space-y-3">
              {optimizations.map((o, i) => (
                <div key={o.name} className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded-full bg-amber-50 text-amber-700 border border-slate-900 text-xl font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                    {i + 1}
                  </span>
                  <div>
                    <div className="text-2xl font-bold text-slate-800">{o.name}</div>
                    <div className="text-xl text-slate-700">{o.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass">
            <h3 className="text-2xl uppercase tracking-wider text-purple-600 font-extrabold mb-3 flex items-center gap-2">
              <Play size={15} /> Live Demo & Tools
            </h3>
            <div className="space-y-2">
              <a href="/demo/" target="_blank" rel="noopener" className="ext-link text-xl flex items-center gap-2">
                <ExternalLink size={13} /> ASR Demo (Streamlit)
              </a>
              <a href="http://localhost:6900" target="_blank" rel="noopener" className="ext-link text-xl flex items-center gap-2">
                <ExternalLink size={13} /> Argilla (Dataset Review)
              </a>
              <a href="http://localhost:5050" target="_blank" rel="noopener" className="ext-link text-xl flex items-center gap-2">
                <ExternalLink size={13} /> MLflow Dashboard
              </a>
              <a href="http://localhost:9001" target="_blank" rel="noopener" className="ext-link text-xl flex items-center gap-2">
                <ExternalLink size={13} /> MinIO (Model Storage)
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

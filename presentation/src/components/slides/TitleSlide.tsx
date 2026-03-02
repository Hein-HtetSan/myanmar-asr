"use client";

import { motion } from "framer-motion";
import type { SlideProps } from "../Presentation";

export default function TitleSlide(_props: SlideProps) {
  return (
    <div className="slide items-center justify-center text-center">
      {/* decorative gradient blobs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-blue-100 rounded-full blur-[100px] opacity-50" />
        <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-emerald-100 rounded-full blur-[100px] opacity-50" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative z-10 flex flex-col items-center gap-5"
      >
        {/* badge */}
        <span className="badge bg-accent/10 text-accent border border-accent/20 text-xl">
          Research Presentation
        </span>

        <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold leading-tight max-w-5xl text-slate-900">
          Myanmar Automatic Speech{" "}
          <span className="bg-gradient-to-r from-accent to-emerald-600 bg-clip-text text-transparent">
            Recognition System
          </span>
        </h1>

        <p className="text-xl md:text-2xl text-slate-700 max-w-2xl">
          Using Fine-Tuned Transformer Models
        </p>

        <p className="text-2xl md:text-xl text-slate-600 max-w-2xl leading-relaxed mt-2">
          Comparative Fine-Tuning of Multilingual Speech Models
          <br />
          for Low-Resource Myanmar ASR
        </p>

        <div className="mt-6 flex flex-col items-center gap-2 text-2xl">
          <span className="text-slate-800 font-extrabold text-xl">
            Hein Htet San
          </span>
          <span className="text-slate-800 font-extrabold text-xl">
            Ye Myat Kyaw
          </span>
          <span className="text-slate-700">March 2026</span>
        </div>

        {/* model pills */}
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          {[
            { name: "Whisper v3 Turbo", color: "bg-red-50 text-red-700 border-slate-900" },
            { name: "Dolphin (Whisper-large-v2)", color: "bg-blue-50 text-blue-700 border-slate-900" },
            { name: "SeamlessM4T v2 Large", color: "bg-emerald-50 text-emerald-700 border-slate-900" },
          ].map((m) => (
            <span key={m.name} className={`badge border ${m.color}`}>
              {m.name}
            </span>
          ))}
        </div>
      </motion.div>

      {/* subtle hint */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="absolute bottom-12 text-xl text-slate-700"
      >
        Press &rarr; or swipe to navigate
      </motion.p>
    </div>
  );
}

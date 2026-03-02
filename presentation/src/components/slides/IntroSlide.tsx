"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Globe, Lightbulb } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
};
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function IntroSlide(_props: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title">Introduction</h2>

      <motion.div
        variants={stagger}
        initial="hidden"
        animate="show"
        className="grid md:grid-cols-3 gap-5 flex-1"
      >
        {/* Background */}
        <motion.div variants={item} className="glass flex flex-col gap-4">
          <div className="flex items-center gap-2 text-accent font-extrabold text-2xl uppercase tracking-wider">
            <Globe size={16} /> Background
          </div>
          <ul className="space-y-3 text-2xl md:text-xl text-slate-600 leading-relaxed">
            <li>
              Myanmar (Burmese) is a{" "}
              <span className="text-amber-600 font-extrabold">
                low-resource language
              </span>{" "}
              with limited ASR research and public datasets
            </li>
            <li>
              Global ASR systems (Google, OpenAI Whisper) perform poorly on Myanmar due to:
            </li>
            <li className="pl-4 text-2xl text-slate-700">
              &bull; Complex script with consonant clusters, stacked characters, and{" "}
              <span className="text-amber-600 font-extrabold">no word boundaries</span>
            </li>
            <li className="pl-4 text-2xl text-slate-700">
              &bull; Very limited training data vs. English (thousands vs millions of hours)
            </li>
            <li className="pl-4 text-2xl text-slate-700">
              &bull; Tonal language with 4 tones affecting meaning
            </li>
          </ul>
        </motion.div>

        {/* Problem */}
        <motion.div variants={item} className="glass flex flex-col gap-4">
          <div className="flex items-center gap-2 text-danger font-extrabold text-2xl uppercase tracking-wider">
            <AlertTriangle size={16} /> Problem Statement
          </div>
          <ul className="space-y-3 text-2xl md:text-xl text-slate-600 leading-relaxed">
            <li>
              No high-quality open-source Myanmar ASR model exists
            </li>
            <li>
              Existing multilingual models achieve{" "}
              <span className="text-red-600 font-bold">WER &gt; 80%</span> on
              Myanmar speech
            </li>
            <li>
              Commercial solutions are expensive and not customizable for local needs
            </li>
          </ul>
        </motion.div>

        {/* Approach */}
        <motion.div variants={item} className="glass flex flex-col gap-4">
          <div className="flex items-center gap-2 text-success font-extrabold text-2xl uppercase tracking-wider">
            <Lightbulb size={16} /> Our Approach
          </div>
          <ul className="space-y-3 text-2xl md:text-xl text-slate-600 leading-relaxed">
            <li>
              Fine-tune{" "}
              <span className="text-emerald-700 font-extrabold">
                3 state-of-the-art
              </span>{" "}
              multilingual speech models on curated Myanmar data
            </li>
            <li>
              Compare architectures:{" "}
              <span className="text-emerald-700 font-extrabold">Whisper v3 Turbo</span>,{" "}
              <span className="text-emerald-700 font-extrabold">Dolphin (Whisper Large-v2)</span>,{" "}
              <span className="text-emerald-700 font-extrabold">SeamlessM4T v2 Large</span>
            </li>
            <li>
              Curate and clean a{" "}
              <span className="text-emerald-700 font-extrabold">54-hour</span>{" "}
              Myanmar dataset from open sources
            </li>
            <li>
              Apply{" "}
              <span className="text-emerald-700 font-extrabold">
                transfer learning with frozen encoder
              </span>{" "}
              strategy to overcome data scarcity
            </li>
          </ul>
        </motion.div>
      </motion.div>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Target } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};
const item = {
  hidden: { opacity: 0, x: -16 },
  show: { opacity: 1, x: 0, transition: { duration: 0.35 } },
};

const primaryObjectives = [
  "Build a high-accuracy Myanmar ASR system by fine-tuning pre-trained multilingual speech models",
  "Curate and clean a Myanmar speech dataset from multiple open-source corpora (~54 hours with augmentation)",
  "Compare 3 model architectures to identify the best approach for low-resource Myanmar ASR",
];

const secondaryObjectives = [
  "Develop a reproducible end-to-end pipeline — from data collection to model evaluation",
  "Implement experiment tracking (MLflow) for systematic model comparison",
  "Establish baseline benchmarks for Myanmar ASR to guide future research",
  "Prepare models for future deployment and inference optimization",
];

const criteria = [
  { label: "CER < 35% on test set", achieved: true, result: "13.04% CER (SeamlessM4T)" },
  { label: "Improvement over zero-shot baselines", achieved: true, result: "WER decreased by 67pp, CER decreased by 75pp" },
  { label: "Complete 3-model comparative analysis", achieved: true, result: "All 3 models trained & evaluated" },
];

export default function ObjectivesSlide(_props: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title">Objectives</h2>

      <div className="grid md:grid-cols-2 gap-6 flex-1">
        {/* left: objectives */}
        <div className="flex flex-col gap-5">
          <div>
            <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-4 flex items-center gap-2">
              <Target size={15} /> Primary Objectives
            </h3>
            <motion.ol
              variants={stagger}
              initial="hidden"
              animate="show"
              className="space-y-3"
            >
              {primaryObjectives.map((o, i) => (
                <motion.li
                  key={i}
                  variants={item}
                  className="flex gap-3 text-2xl md:text-xl text-slate-700 leading-relaxed"
                >
                  <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 text-accent text-xl font-bold flex items-center justify-center mt-0.5">
                    {i + 1}
                  </span>
                  <span>{o}</span>
                </motion.li>
              ))}
            </motion.ol>
          </div>

          <div>
            <h3 className="text-2xl uppercase tracking-wider text-slate-700 font-extrabold mb-3">
              Secondary Objectives
            </h3>
            <motion.ol
              variants={stagger}
              initial="hidden"
              animate="show"
              className="space-y-2"
            >
              {secondaryObjectives.map((o, i) => (
                <motion.li
                  key={i}
                  variants={item}
                  className="flex gap-3 text-2xl text-slate-600 leading-relaxed"
                >
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-slate-100 text-slate-700 text-xl font-bold flex items-center justify-center mt-0.5">
                    {i + 4}
                  </span>
                  <span>{o}</span>
                </motion.li>
              ))}
            </motion.ol>
          </div>
        </div>

        {/* right: success criteria */}
        <div>
          <h3 className="text-2xl uppercase tracking-wider text-success font-extrabold mb-4 flex items-center gap-2">
            <CheckCircle2 size={15} /> Success Criteria
          </h3>
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="show"
            className="space-y-4"
          >
            {criteria.map((c, i) => (
              <motion.div
                key={i}
                variants={item}
                className="glass flex items-start gap-3"
              >
                <CheckCircle2
                  size={20}
                  className={c.achieved ? "text-success mt-0.5" : "text-slate-700 mt-0.5"}
                />
                <div>
                  <div className="font-bold text-2xl md:text-xl text-slate-800">{c.label}</div>
                  <div className="text-xl md:text-2xl text-emerald-600 font-mono mt-1">
                    {c.result}
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
}

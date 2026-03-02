"use client";

import { motion } from "framer-motion";
import { Wrench, ExternalLink } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.04 } } };
const item = { hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0, transition: { duration: 0.25 } } };

const toolCategories = [
  {
    title: "ML / Training",
    color: "text-red-600",
    tools: [
      { name: "PyTorch 2.5.1", desc: "Deep learning framework" },
      { name: "Transformers 5.2.0", desc: "Hugging Face model library" },
      { name: "CUDA 12.1", desc: "NVIDIA GPU acceleration" },
      { name: "Datasets", desc: "Hugging Face data loading" },
      { name: "Evaluate", desc: "WER / CER metric computation" },
    ],
  },
  {
    title: "Experiment Tracking",
    color: "text-blue-600",
    tools: [
      { name: "MLflow 2.19", desc: "Experiment & model tracking", url: "http://localhost:5050" },
      { name: "MinIO", desc: "S3-compatible artifact store", url: "http://localhost:9001" },
      { name: "autossh", desc: "Persistent SSH tunnel for logging" },
    ],
  },
  {
    title: "Data / Annotation",
    color: "text-emerald-600",
    tools: [
      { name: "Label Studio", desc: "Audio annotation platform", url: "http://localhost:8081" },
      { name: "Argilla", desc: "ML dataset review & curation", url: "http://localhost:6900" },
      { name: "librosa / soundfile", desc: "Audio processing" },
      { name: "Unicode NFC", desc: "Myanmar text normalization" },
    ],
  },
  {
    title: "Infrastructure",
    color: "text-purple-600",
    tools: [
      { name: "Docker Compose", desc: "Multi-service orchestration" },
      { name: "Vast.ai RTX 4090", desc: "GPU cloud training" },
      { name: "FastAPI", desc: "Cloud inference API server" },
      { name: "nginx", desc: "Reverse proxy" },
    ],
  },
  {
    title: "Demo / Presentation",
    color: "text-amber-600",
    tools: [
      { name: "Streamlit", desc: "Interactive ASR demo app", url: "/demo/" },
      { name: "Next.js 15", desc: "Slide presentation framework" },
      { name: "Framer Motion", desc: "Slide animations" },
      { name: "Jupyter Lab", desc: "Data exploration", url: "http://localhost:8888/?token=myanmar123" },
    ],
  },
  {
    title: "Data Sources",
    color: "text-slate-600",
    tools: [
      { name: "FLEURS (my_mm)", desc: "Google multilingual speech" },
      { name: "OpenSLR-80", desc: "Myanmar read speech corpus" },
      { name: "Common Voice", desc: "Mozilla crowd-sourced audio" },
      { name: "VOA Burmese", desc: "News broadcast audio" },
    ],
  },
];

export default function ToolsSlide(_props: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title flex items-center gap-3">
        <Wrench size={24} /> Tools &amp; Infrastructure
      </h2>

      <motion.div
        variants={stagger}
        initial="hidden"
        animate="show"
        className="grid grid-cols-2 md:grid-cols-3 gap-4 flex-1"
      >
        {toolCategories.map((cat) => (
          <motion.div key={cat.title} variants={item} className="glass">
            <h3 className={`text-xl uppercase tracking-wider font-extrabold mb-2 ${cat.color}`}>
              {cat.title}
            </h3>
            <div className="space-y-1.5">
              {cat.tools.map((t) => (
                <div key={t.name} className="text-lg">
                  {"url" in t && t.url ? (
                    <a href={t.url} target="_blank" rel="noopener" className="ext-link text-lg">
                      <ExternalLink size={10} /> <span className="font-bold text-slate-800">{t.name}</span>
                    </a>
                  ) : (
                    <span className="font-bold text-slate-800">{t.name}</span>
                  )}
                  <span className="text-slate-600 ml-1">— {t.desc}</span>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}

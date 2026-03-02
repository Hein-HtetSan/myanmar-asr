"use client";

import { motion } from "framer-motion";
import { Database, BarChart3 } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.08 } } };
const item = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } };

const sources = [
  { name: "Google FLEURS", type: "Read speech", origin: "Crowd-sourced", samples: "~3,000" },
  { name: "OpenSLR-80", type: "Read speech", origin: "Crowd-sourced", samples: "~5,000" },
  { name: "YODAS (YouTube)", type: "Spontaneous speech", origin: "Web-scraped", samples: "~9,000" },
  { name: "Speed Augmentation", type: "0.9x + 1.1x speed", origin: "Generated from clean data", samples: "+10,515" },
];

const splits = [
  { split: "Train", samples: "20,814", hours: "~49.5h" },
  { split: "Validation", samples: "639", hours: "~1.5h" },
  { split: "Test", samples: "1,252", hours: "~3.2h" },
  { split: "Total", samples: "22,705", hours: "~54.2h" },
];

export default function DatasetSlide({ onImageClick }: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title">Dataset Description</h2>

      <div className="grid md:grid-cols-5 gap-5 flex-1">
        {/* left: tables */}
        <div className="md:col-span-3 flex flex-col gap-5">
          {/* sources */}
          <div className="glass flex-1">
            <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-3 flex items-center gap-2">
              <Database size={15} /> Data Sources (All Open-Source)
            </h3>
            <motion.table variants={stagger} initial="hidden" animate="show" className="data-table">
              <thead>
                <tr><th>Source</th><th>Type</th><th>Origin</th><th>Samples</th></tr>
              </thead>
              <tbody>
                {sources.map((s) => (
                  <motion.tr key={s.name} variants={item}>
                    <td className="font-bold text-slate-800">{s.name}</td>
                    <td>{s.type}</td>
                    <td className="text-slate-700">{s.origin}</td>
                    <td className="font-mono text-accent">{s.samples}</td>
                  </motion.tr>
                ))}
              </tbody>
            </motion.table>
          </div>

          {/* splits */}
          <div className="glass">
            <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-3 flex items-center gap-2">
              <BarChart3 size={15} /> Final Dataset Statistics
            </h3>
            <div className="flex gap-4 flex-wrap">
              {splits.map((s) => (
                <div
                  key={s.split}
                  className={`flex-1 min-w-[110px] rounded-xl p-3 text-center ${
                    s.split === "Total"
                      ? "bg-accent/5 border-2 border-accent/20"
                      : "bg-slate-50 border border-slate-200"
                  }`}
                >
                  <div className="text-xl text-slate-700 uppercase tracking-wider">{s.split}</div>
                  <div className={`text-xl md:text-2xl font-bold mt-1 tabular-nums ${s.split === "Total" ? "text-accent" : "text-slate-800"}`}>{s.samples}</div>
                  <div className="text-xl text-slate-700 mt-0.5">{s.hours}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* right: chart + characteristics */}
        <div className="md:col-span-2 flex flex-col gap-5">
          <div className="glass flex-1 flex flex-col">
            <h3 className="text-2xl uppercase tracking-wider text-slate-700 font-extrabold mb-3">
              Dataset Overview
            </h3>
            <img
              src="/charts/01_overview_dashboard.png"
              alt="Dataset Overview Dashboard"
              className="chart-thumb rounded-lg flex-1 object-contain"
              onClick={() => onImageClick("/charts/01_overview_dashboard.png", "Dataset Overview Dashboard")}
            />
          </div>
          <div className="glass text-2xl text-slate-600 space-y-1.5">
            <h3 className="text-xl uppercase tracking-wider text-slate-700 font-extrabold mb-2">Key Characteristics</h3>
            <div><span className="text-slate-800 font-bold">Sample rate:</span> 16,000 Hz (mono)</div>
            <div><span className="text-slate-800 font-bold">Language:</span> Myanmar (Burmese) — <code className="text-accent text-xl bg-accent/5 px-1 rounded">my</code></div>
            <div><span className="text-slate-800 font-bold">Script:</span> Myanmar Unicode (NFC)</div>
            <div><span className="text-slate-800 font-bold">Avg. duration:</span> ~8.6 seconds per sample</div>
            <div><span className="text-slate-800 font-bold">Duration range:</span> 0.5 – 30 seconds</div>
          </div>
        </div>
      </div>
    </div>
  );
}

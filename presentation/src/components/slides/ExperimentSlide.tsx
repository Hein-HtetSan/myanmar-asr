"use client";

import { motion } from "framer-motion";
import { Cpu, ExternalLink, Wrench, KeyRound } from "lucide-react";

type SlideProps = import("../Presentation").SlideProps;

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.06 } } };
const item = { hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0, transition: { duration: 0.3 } } };

const configs = [
  { param: "Base model", turbo: "whisper-large-v3-turbo", dolphin: "whisper-large-v2", seamless: "seamless-m4t-v2-large" },
  { param: "Strategy", turbo: "Full model (decoder emphasis)", dolphin: "Frozen encoder, decoder-only", seamless: "Frozen speech encoder, adaptor + text decoder" },
  { param: "Learning rate", turbo: "1e-5", dolphin: "1e-4", seamless: "5e-5" },
  { param: "LR scheduler", turbo: "Linear", dolphin: "Cosine annealing", seamless: "Cosine annealing" },
  { param: "Warmup", turbo: "500 steps", dolphin: "6% of total steps", seamless: "8% of total steps" },
  { param: "Weight decay", turbo: "0.0", dolphin: "0.05", seamless: "0.05" },
  { param: "Label smoothing", turbo: "0.0", dolphin: "0.1", seamless: "0.1" },
  { param: "Eff. batch size", turbo: "32 (4×8 accum)", dolphin: "24 (6×4 accum)", seamless: "32 (4×8 accum)" },
  { param: "Max epochs", turbo: "5", dolphin: "15", seamless: "12" },
  { param: "Precision", turbo: "fp16", dolphin: "bf16", seamless: "bf16" },
  { param: "Eval frequency", turbo: "Every 500 steps", dolphin: "Every 243 steps", seamless: "Every 182 steps" },
  { param: "Early stopping", turbo: "Patience = 3", dolphin: "Patience = 5", seamless: "Patience = 5" },
  { param: "Grad. checkpoint", turbo: "Yes", dolphin: "Yes", seamless: "Yes" },
];

const challenges = [
  { title: "Transformers 5.x breaking change", desc: "Implemented manual _shift_tokens_right() for decoder input" },
  { title: "SeamlessM4T overflow error", desc: "np.clip(pred_ids, 0, vocab_size - 1) during eval" },
  { title: "SSH tunnel instability", desc: "Solved with autossh for persistent MLflow logging" },
  { title: "WER > 100% early training", desc: "Normal for Myanmar — CER used as primary metric" },
];

export default function ExperimentSlide(_props: SlideProps) {
  return (
    <div className="slide">
      <h2 className="slide-title">Experimental Setup</h2>

      <div className="grid md:grid-cols-5 gap-8 flex-1">
        {/* config table */}
        <div className="md:col-span-3 glass overflow-x-auto">
          <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-3">
            Training Configurations — All 3 Models
          </h3>
          <motion.table
            variants={stagger}
            initial="hidden"
            animate="show"
            className="data-table text-xl md:text-2xl"
          >
            <thead>
              <tr>
                <th>Parameter</th>
                <th className="text-red-600">Turbo</th>
                <th className="text-blue-600">Dolphin</th>
                <th className="text-emerald-600">Seamless</th>
              </tr>
            </thead>
            <tbody>
              {configs.map((c) => (
                <motion.tr key={c.param} variants={item}>
                  <td className="font-bold text-slate-700">{c.param}</td>
                  <td className="font-mono text-slate-600">{c.turbo}</td>
                  <td className="font-mono text-slate-600">{c.dolphin}</td>
                  <td className="font-mono text-slate-600">{c.seamless}</td>
                </motion.tr>
              ))}
            </tbody>
          </motion.table>
        </div>

        {/* right: hardware + challenges + links */}
        <div className="md:col-span-2 flex flex-col gap-4">
          {/* hardware */}
          <div className="glass">
            <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-3 flex items-center gap-2">
              <Cpu size={15} /> Hardware &amp; Infrastructure
            </h3>
            <div className="space-y-2 text-2xl text-slate-700">
              <div className="flex items-center gap-2">
                <span className="badge bg-green-50 text-green-700 border border-green-200">GPU</span>
                <span>NVIDIA RTX 4090 (24 GB) — Vast.ai</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="badge bg-purple-50 text-purple-700 border border-purple-200">Time</span>
                <span>~12.2h total (159 + 335 + 239 min)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="badge bg-blue-50 text-blue-700 border border-blue-200">Stack</span>
                <span>Transformers 5.2.0 · PyTorch 2.5.1 + CUDA 12.1</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="badge bg-amber-50 text-amber-700 border border-amber-200">Track</span>
                <span>MLflow (self-hosted with Docker)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="badge bg-slate-100 text-slate-600 border border-slate-200">SSH</span>
                <span>autossh reverse tunnel for persistent MLflow logging</span>
              </div>
            </div>
          </div>

          {/* Service Credentials & Links */}
          <div className="glass flex-1">
            <h3 className="text-2xl uppercase tracking-wider text-accent font-extrabold mb-3 flex items-center gap-2">
              <KeyRound size={15} /> Service Logins
            </h3>
            <div className="space-y-1.5 text-lg">
              {[
                { name: "MLflow", url: "http://localhost:5050", user: "", pw: "", note: "No auth" },
                { name: "Label Studio", url: "http://localhost:8081", user: "admin@myanmar-asr.local", pw: "myanmar123" },
                { name: "Argilla", url: "http://localhost:6900", user: "argilla", pw: "myanmar123" },
                { name: "MinIO Console", url: "http://localhost:9001", user: "minioadmin", pw: "minioadmin123" },
                { name: "Jupyter Lab", url: "http://localhost:8888/?token=myanmar123", user: "", pw: "", note: "Token in URL" },
              ].map((s) => (
                <div key={s.name} className="flex items-center gap-2 flex-wrap">
                  <a href={s.url} target="_blank" rel="noopener" className="ext-link text-lg min-w-[130px]">
                    <ExternalLink size={11} /> {s.name}
                  </a>
                  {s.note ? (
                    <span className="font-mono text-slate-400 text-base">{s.note}</span>
                  ) : (
                    <span className="font-mono text-slate-500 text-base">
                      {s.user} / {s.pw}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* challenges */}
          <div className="glass flex-1">
            <h3 className="text-2xl uppercase tracking-wider text-amber-600 font-extrabold mb-3 flex items-center gap-2">
              <Wrench size={15} /> Technical Challenges Solved
            </h3>
            <div className="space-y-2.5">
              {challenges.map((c) => (
                <div key={c.title} className="text-xl md:text-2xl">
                  <div className="font-bold text-slate-800">{c.title}</div>
                  <div className="text-slate-700">{c.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

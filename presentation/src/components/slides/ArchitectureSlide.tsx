"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Lock, Unlock } from "lucide-react";
import type { SlideProps } from "../Presentation";

const models = [
  {
    id: "turbo",
    name: "Whisper v3 Turbo",
    base: "openai/whisper-large-v3-turbo",
    params: "809M",
    color: "border-slate-900 bg-red-50",
    accent: "text-red-700",
    badge: "bg-red-50 text-red-700",
    arch: "Encoder-Decoder Transformer (Distilled)",
    encoder: { name: "Whisper Encoder", params: "637M", frozen: false },
    decoder: { name: "Distilled Decoder (4 layers)", params: "172M", trainable: true },
    note: "Distilled from 32 ➞ 4 decoder layers. Full model fine-tuned (no explicit freeze). Fastest inference due to shallow decoder.",
  },
  {
    id: "dolphin",
    name: "Dolphin (Whisper-large-v2)",
    base: "openai/whisper-large-v2",
    params: "1.5B",
    color: "border-slate-900 bg-blue-50",
    accent: "text-blue-700",
    badge: "bg-blue-50 text-blue-700",
    arch: "Encoder-Decoder Transformer",
    encoder: { name: "Whisper Encoder", params: "~637M", frozen: true },
    decoder: { name: "Full Decoder (32 layers)", params: "~300M", trainable: true },
    note: "Frozen encoder. Decoder-only fine-tuning with cosine LR + label smoothing. 32-layer decoder enables deep language modeling.",
  },
  {
    id: "seamless",
    name: "SeamlessM4T v2 Large",
    base: "facebook/seamless-m4t-v2-large",
    params: "2.3B",
    color: "border-slate-900 bg-emerald-50",
    accent: "text-emerald-700",
    badge: "bg-emerald-50 text-emerald-700",
    arch: "Encoder + Length Adaptor + Decoder",
    encoder: { name: "w2v-BERT 2.0 Speech Encoder", params: "~600M", frozen: true },
    decoder: { name: "Text Decoder + Length Adaptor", params: "~400M", trainable: true },
    note: "Frozen speech encoder. Length adaptor compresses speech representations ➞ text decoder generates Myanmar text. Outstanding character accuracy.",
  },
];

export default function ArchitectureSlide(_props: SlideProps) {
  const [sel, setSel] = useState(1); // default Dolphin

  const m = models[sel];

  return (
    <div className="slide">
      <h2 className="slide-title">Model Architecture</h2>

      <div className="flex flex-col flex-1 gap-5">
        {/* tabs */}
        <div className="flex gap-2">
          {models.map((mod, i) => (
            <button
              key={mod.id}
              onClick={() => setSel(i)}
              className={`flex-1 rounded-xl px-4 py-3 text-2xl font-bold border transition-all ${
                i === sel
                  ? `${mod.color} border-opacity-100 shadow-sm`
                  : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700"
              }`}
            >
              <div className={i === sel ? mod.accent : ""}>{mod.name}</div>
              <div className="text-xl text-slate-700 mt-0.5">{mod.params} params</div>
            </button>
          ))}
        </div>

        {/* detail */}
        <AnimatePresence mode="wait">
          <motion.div
            key={m.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
            className="flex-1 grid md:grid-cols-2 gap-5"
          >
            {/* diagram */}
            <div className={`glass border ${m.color} flex flex-col gap-4`}>
              <div className={`text-xl uppercase tracking-wider font-extrabold ${m.accent}`}>
                {m.arch}
              </div>

              <div className="flex gap-3 flex-1">
                {/* encoder block */}
                <div className="flex-1 rounded-xl border border-slate-200 bg-slate-50 p-4 flex flex-col gap-2">
                  <div className="flex items-center gap-2 text-2xl font-extrabold">
                    {m.encoder.frozen ? (
                      <Lock size={14} className="text-amber-600" />
                    ) : (
                      <Unlock size={14} className="text-emerald-600" />
                    )}
                    <span className={m.encoder.frozen ? "text-amber-700" : "text-emerald-700"}>
                      {m.encoder.frozen ? "FROZEN" : "TRAINABLE"}
                    </span>
                  </div>
                  <div className="text-xl font-bold text-slate-800">{m.encoder.name}</div>
                  <div className="text-2xl text-slate-700 font-mono">{m.encoder.params}</div>
                  <div className="mt-auto text-xl text-slate-700">
                    {m.encoder.frozen
                      ? "Pre-trained acoustic features preserved"
                      : "Full model fine-tuning"}
                  </div>
                </div>

                {/* arrow */}
                <div className="flex items-center text-slate-700 text-xl">➞</div>

                {/* decoder block */}
                <div className="flex-1 rounded-xl border-2 border-accent/30 bg-accent/5 p-4 flex flex-col gap-2">
                  <div className="flex items-center gap-2 text-2xl font-extrabold text-accent">
                    <Layers size={14} /> TRAINABLE
                  </div>
                  <div className="text-xl font-bold text-slate-800">{m.decoder.name}</div>
                  <div className="text-2xl text-slate-700 font-mono">{m.decoder.params}</div>
                  <div className="mt-auto text-xl text-slate-700">
                    Learns Myanmar-specific text generation
                  </div>
                </div>
              </div>

              <div className="text-xl text-slate-700 border-t border-slate-200 pt-2">
                Input: 16 kHz audio ➞ Log-Mel Spectrogram ➞ Output: Myanmar text tokens
              </div>
            </div>

            {/* info panel */}
            <div className="flex flex-col gap-4">
              <div className="glass">
                <div className="text-xl text-slate-700 uppercase tracking-wider mb-2">Base Model</div>
                <code className={`text-2xl font-mono ${m.accent}`}>{m.base}</code>
              </div>
              <div className="glass flex-1">
                <div className="text-xl text-slate-700 uppercase tracking-wider mb-2">Key Notes</div>
                <p className="text-2xl text-slate-600 leading-relaxed">{m.note}</p>
              </div>
              <div className="glass">
                <div className="text-xl text-slate-700 uppercase tracking-wider mb-2">Why Frozen Encoder?</div>
                <p className="text-2xl text-slate-600 leading-relaxed">
                  <span className="text-amber-600 font-extrabold">Frozen encoder</span> preserves
                  universal acoustic features learned from massive multilingual data. With only ~54h
                  of Myanmar data, retraining the encoder risks overfitting.
                  Training only the decoder is <span className="font-extrabold text-slate-800">faster and more memory-efficient</span>.
                  The decoder learns Myanmar script, grammar, and vocabulary.
                </p>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

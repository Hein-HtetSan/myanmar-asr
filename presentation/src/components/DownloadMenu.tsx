"use client";

import { useState, useRef, useEffect } from "react";
import { Download, FileText, Presentation, X } from "lucide-react";

/* ── slide text data for PPTX export ── */
const slideData = [
  { title: "Myanmar ASR System", body: "Comparative Fine-Tuning of Multilingual Speech Models\nfor Low-Resource Myanmar ASR\n\nHein Htet San — March 2026" },
  { title: "Introduction", body: "• Myanmar is a low-resource language with limited ASR data\n• Existing multilingual models achieve WER > 80%\n• Approach: fine-tune 3 pre-trained models on curated 54h dataset" },
  { title: "Objectives", body: "1. Build high-accuracy Myanmar ASR via fine-tuning\n2. Curate 54h dataset from open-source corpora\n3. Compare 3 architectures\n✅ CER < 35% achieved (13.04%)\n✅ All 3 models completed" },
  { title: "Dataset", body: "Sources: FLEURS + OpenSLR-80 + YODAS + augmentation\n22,705 samples | 54.2 hours\nTrain: 20,814 | Val: 639 | Test: 1,252" },
  { title: "Preprocessing", body: "Raw 17,552 → Clean 12,190 (30% removed)\n+ Speed augmentation (0.9x, 1.1x)\n→ Final: 22,705 samples\nSteps: audio validation, duration filter, text validation, SNR filter, dedup, normalization" },
  { title: "Model Architecture", body: "Whisper v3 Turbo — 809M params (distilled decoder)\nDolphin (Whisper-large-v2) — 1.5B params (frozen encoder)\nSeamlessM4T v2 Large — 2.3B params (frozen speech encoder + adaptor)" },
  { title: "Experimental Setup", body: "GPU: RTX 4090 (24GB) on Vast.ai\nFramework: HuggingFace Transformers 5.2.0\nTotal training: ~12.2 hours\nTracking: MLflow (self-hosted)" },
  { title: "Results", body: "Dolphin: WER 33.02% | CER 28.00% — BEST WER\nSeamless: WER 49.12% | CER 13.04% — BEST CER\nTurbo: WER 54.49% | CER 36.00%\nImprovement: WER ↓67pp, CER ↓75pp from baseline" },
  { title: "Deployment Plan", body: "Word-level tasks → Dolphin (best WER)\nCharacter-level tasks → SeamlessM4T (best CER)\nLow-latency tasks → Whisper Turbo\nOptimization: CTranslate2, quantization, ONNX" },
  { title: "Conclusion", body: "✅ All 3 models fine-tuned successfully\n✅ CER 13.04% (SeamlessM4T) — exceeds 35% target\n✅ WER 33.02% (Dolphin) — best word accuracy\nKey: frozen encoder + cosine LR + label smoothing\nFuture: model optimization, web demo, dataset expansion" },
];

export default function DownloadMenu() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const handlePDF = () => {
    setOpen(false);
    window.print();
  };

  const handlePPTX = async () => {
    setOpen(false);
    const PptxGenJS = (await import("pptxgenjs")).default;
    const pptx = new PptxGenJS();
    pptx.defineLayout({ name: "WIDE", width: 13.33, height: 7.5 });
    pptx.layout = "WIDE";
    pptx.author = "Hein Htet San";
    pptx.title = "Myanmar ASR Research Presentation";

    for (const s of slideData) {
      const slide = pptx.addSlide();
      slide.background = { color: "FFFFFF" };
      slide.addText(s.title, {
        x: 0.8, y: 0.5, w: 11.7, h: 1,
        fontSize: 32, color: "1E293B", bold: true,
        fontFace: "Helvetica",
      });
      slide.addShape(pptx.ShapeType.rect, {
        x: 0.8, y: 1.4, w: 2.5, h: 0.04, fill: { color: "2563EB" },
      });
      slide.addText(s.body, {
        x: 0.8, y: 1.8, w: 11.7, h: 5,
        fontSize: 16, color: "475569",
        fontFace: "Helvetica", lineSpacing: 26,
        valign: "top",
      });
    }

    pptx.writeFile({ fileName: "Myanmar_ASR_Presentation.pptx" });
  };

  return (
    <div ref={ref} className="no-print fixed bottom-5 right-5 z-50">
      {/* toggle */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-12 h-12 rounded-full bg-accent hover:bg-accent-light
                   transition-colors shadow-lg shadow-accent/30
                   flex items-center justify-center text-black"
        aria-label="Download"
      >
        {open ? <X size={20} /> : <Download size={20} />}
      </button>

      {/* dropdown */}
      {open && (
        <div className="absolute bottom-14 right-0 w-52 rounded-xl border border-slate-200 bg-white shadow-xl py-2 animate-fade-in">
          <button
            onClick={handlePDF}
            className="w-full flex items-center gap-3 px-4 py-2.5 text-2xl text-slate-700 hover:bg-slate-50 transition-colors text-left"
          >
            <FileText size={16} className="text-red-500" />
            <span>Save as PDF</span>
          </button>
          <button
            onClick={handlePPTX}
            className="w-full flex items-center gap-3 px-4 py-2.5 text-2xl text-slate-700 hover:bg-slate-50 transition-colors text-left"
          >
            <Presentation size={16} className="text-orange-500" />
            <span>Download PPTX</span>
          </button>
        </div>
      )}
    </div>
  );
}

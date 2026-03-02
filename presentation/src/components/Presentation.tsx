"use client";

import { useState, useCallback, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Navigation from "./Navigation";
import DownloadMenu from "./DownloadMenu";
import ImageModal from "./ImageModal";

import TitleSlide from "./slides/TitleSlide";
import IntroSlide from "./slides/IntroSlide";
import ObjectivesSlide from "./slides/ObjectivesSlide";
import DatasetSlide from "./slides/DatasetSlide";
import PreprocessingSlide from "./slides/PreprocessingSlide";
import ArchitectureSlide from "./slides/ArchitectureSlide";
import ToolsSlide from "./slides/ToolsSlide";
import ExperimentSlide from "./slides/ExperimentSlide";
import ResultsSlide from "./slides/ResultsSlide";
import DeploymentSlide from "./slides/DeploymentSlide";
import ConclusionSlide from "./slides/ConclusionSlide";
import ThankYouSlide from "./slides/ThankYouSlide";

export interface SlideProps {
  onImageClick: (src: string, alt: string) => void;
}

const SLIDES = [
  TitleSlide,
  IntroSlide,
  ObjectivesSlide,
  DatasetSlide,
  PreprocessingSlide,
  ArchitectureSlide,
  ToolsSlide,
  ExperimentSlide,
  ResultsSlide,
  DeploymentSlide,
  ConclusionSlide,
  ThankYouSlide,
];

const SLIDE_NAMES = [
  "Title",
  "Introduction",
  "Objectives",
  "Dataset",
  "Preprocessing",
  "Architecture",
  "Tools",
  "Experiment",
  "Results",
  "Deployment",
  "Conclusion",
  "Thank You",
];

export default function Presentation() {
  const [cur, setCur] = useState(0);
  const [dir, setDir] = useState(1);
  const [modal, setModal] = useState<{ src: string; alt: string } | null>(null);

  const goTo = useCallback(
    (i: number) => {
      if (i < 0 || i >= SLIDES.length || i === cur) return;
      setDir(i > cur ? 1 : -1);
      setCur(i);
    },
    [cur],
  );

  const next = useCallback(() => goTo(cur + 1), [cur, goTo]);
  const prev = useCallback(() => goTo(cur - 1), [cur, goTo]);

  /* keyboard */
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (modal) {
        if (e.key === "Escape") setModal(null);
        return;
      }
      if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown")
        next();
      if (e.key === "ArrowLeft" || e.key === "PageUp") prev();
      if (e.key === "Home") goTo(0);
      if (e.key === "End") goTo(SLIDES.length - 1);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [next, prev, goTo, modal]);

  /* touch */
  useEffect(() => {
    let x0 = 0;
    const start = (e: TouchEvent) => (x0 = e.touches[0].clientX);
    const end = (e: TouchEvent) => {
      const dx = e.changedTouches[0].clientX - x0;
      if (Math.abs(dx) > 60) dx < 0 ? next() : prev();
    };
    window.addEventListener("touchstart", start, { passive: true });
    window.addEventListener("touchend", end, { passive: true });
    return () => {
      window.removeEventListener("touchstart", start);
      window.removeEventListener("touchend", end);
    };
  }, [next, prev]);

  const Slide = SLIDES[cur];

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-bg">
      {/* progress bar */}
      <div className="absolute top-0 left-0 right-0 h-[3px] bg-slate-100 z-50 no-print">
        <motion.div
          className="h-full bg-gradient-to-r from-accent to-blue-400"
          animate={{ width: `${((cur + 1) / SLIDES.length) * 100}%` }}
          transition={{ duration: 0.35 }}
        />
      </div>

      {/* slide */}
      <AnimatePresence mode="wait" custom={dir}>
        <motion.div
          key={cur}
          custom={dir}
          initial={{ opacity: 0, x: dir * 60 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: dir * -60 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="absolute inset-0"
        >
          <Slide
            onImageClick={(src, alt) => setModal({ src, alt })}
          />
        </motion.div>
      </AnimatePresence>

      {/* navigation */}
      <Navigation
        current={cur}
        total={SLIDES.length}
        names={SLIDE_NAMES}
        onGoTo={goTo}
        onPrev={prev}
        onNext={next}
      />

      {/* download */}
      <DownloadMenu />

      {/* image modal */}
      <AnimatePresence>
        {modal && (
          <ImageModal
            src={modal.src}
            alt={modal.alt}
            onClose={() => setModal(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

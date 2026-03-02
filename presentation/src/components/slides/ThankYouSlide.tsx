"use client";

import { motion } from "framer-motion";
import { Github, Mail, Sparkles, BookOpen } from "lucide-react";
import type { SlideProps } from "../Presentation";

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.1 } } };
const item = { hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0, transition: { duration: 0.35 } } };

export default function ThankYouSlide(_props: SlideProps) {
  return (
    <div className="slide flex flex-col items-center justify-center text-center">
      <div className="w-full max-w-4xl glass bg-white !p-12 border-4 border-slate-900 shadow-[8px_8px_0_0_rgba(15,23,42,1)] relative overflow-hidden">
        
        {/* Decorative corner blocks */}
        <div className="absolute -top-12 -right-12 w-24 h-24 bg-emerald-400 rotate-45 border-b-4 border-slate-900 shadow-sm"></div>
        <div className="absolute -bottom-12 -left-12 w-24 h-24 bg-blue-400 rotate-45 border-t-4 border-slate-900 shadow-sm"></div>

        <motion.div
          variants={stagger}
          initial="hidden"
          animate="show"
          className="flex flex-col gap-8 relative z-10"
        >
          <motion.div variants={item}>
            <div className="flex justify-center mb-6">
               <Sparkles className="text-emerald-500 w-16 h-16" strokeWidth={1.5} />
            </div>
            <h1 className="text-5xl md:text-7xl font-black text-slate-900 uppercase tracking-tighter mb-4">
              Thank You!
            </h1>
            <div className="h-2 w-32 bg-slate-900 mx-auto mb-6"></div>
            <p className="text-2xl md:text-3xl text-slate-700 font-bold max-w-2xl mx-auto leading-relaxed">
              Open-Source Myanmar ASR System
            </p>
          </motion.div>

          <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <a href="https://github.com/Hein-HtetSan/myanmar-asr" target="_blank" rel="noreferrer" className="flex flex-col items-center justify-center gap-3 p-6 border-4 border-slate-900 bg-emerald-50 hover:bg-emerald-100 hover:shadow-[6px_6px_0_0_rgba(15,23,42,1)] hover:-translate-y-1 hover:-translate-x-1 transition-all group">
              <Github size={40} className="text-emerald-700 group-hover:scale-110 transition-transform" />
              <div className="text-xl md:text-2xl font-black text-slate-900 uppercase tracking-wider mt-2">Source Code</div>
              <div className="text-base md:text-lg text-slate-700 font-medium">Dataset, Models & Tools</div>
            </a>
            
            <a href="mailto:contact@heinhtet.san" className="flex flex-col items-center justify-center gap-3 p-6 border-4 border-slate-900 bg-blue-50 hover:bg-blue-100 hover:shadow-[6px_6px_0_0_rgba(15,23,42,1)] hover:-translate-y-1 hover:-translate-x-1 transition-all group">
              <Mail size={40} className="text-blue-700 group-hover:scale-110 transition-transform" />
              <div className="text-xl md:text-2xl font-black text-slate-900 uppercase tracking-wider mt-2">Get In Touch</div>
              <div className="text-base md:text-lg text-slate-700 font-medium">Questions & Collaborations</div>
            </a>
          </motion.div>
          
          <motion.div variants={item} className="mt-6 pt-6 flex items-center justify-center gap-3 text-xl md:text-2xl text-slate-800 font-extrabold border-t-4 border-slate-900 border-dashed">
            <BookOpen className="text-slate-900" />
            Questions & Answers
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}

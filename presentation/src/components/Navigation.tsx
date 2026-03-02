"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

interface Props {
  current: number;
  total: number;
  names: string[];
  onGoTo: (i: number) => void;
  onPrev: () => void;
  onNext: () => void;
}

export default function Navigation({
  current,
  total,
  names,
  onGoTo,
  onPrev,
  onNext,
}: Props) {
  return (
    <>
      {/* side arrows */}
      {current > 0 && (
        <button
          onClick={onPrev}
          className="no-print absolute left-4 top-1/2 -translate-y-1/2 z-40
                     w-12 h-12 flex items-center justify-center
                     bg-white hover:bg-slate-50 border-2 border-slate-900 shadow-[4px_4px_0_0_rgba(15,23,42,1)] transition-transform hover:-translate-y-1 text-slate-900"
          aria-label="Previous slide"
        >
          <ChevronLeft size={28} strokeWidth={3} />
        </button>
      )}
      {current < total - 1 && (
        <button
          onClick={onNext}
          className="no-print absolute right-4 top-1/2 -translate-y-1/2 z-40
                     w-12 h-12 flex items-center justify-center
                     bg-white hover:bg-slate-50 border-2 border-slate-900 shadow-[4px_4px_0_0_rgba(15,23,42,1)] transition-transform hover:-translate-y-1 text-slate-900"
          aria-label="Next slide"
        >
          <ChevronRight size={28} strokeWidth={3} />
        </button>
      )}

      {/* bottom bar: dots + counter */}
      <div className="no-print absolute bottom-0 left-0 right-0 h-16 z-40 flex items-center justify-center gap-1 bg-white border-t-2 border-slate-900 shadow-[0_-4px_0_0_rgba(15,23,42,1)]">
        {/* slide counter (left) */}
        <span className="absolute left-6 text-xl text-slate-700 font-mono tabular-nums font-bold">
          {current + 1}/{total}
        </span>

        {/* dots */}
        <div className="flex items-center gap-2">
          {names.map((name, i) => (
            <button
              key={i}
              onClick={() => onGoTo(i)}
              title={name}
              className={`transition-all duration-200 ${
                i === current
                  ? "w-8 h-3 bg-slate-900 shadow-[2px_2px_0_0_rgba(0,0,0,0.3)]"
                  : "w-3 h-3 bg-slate-300 hover:bg-slate-400 border-2 border-transparent hover:border-slate-900"
              }`}
              style={i === current ? {} : { borderRadius: '50%' }}
              aria-label={`Go to ${name}`}
            />
          ))}
        </div>

        {/* keyboard hint (right) */}
        <span className="absolute right-6 text-xl text-slate-600 hidden md:block font-bold">
          ← → keys
        </span>
      </div>
    </>
  );
}

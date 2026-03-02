import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Myanmar ASR — Research Presentation",
  description:
    "Comparative Fine-Tuning of Multilingual Speech Models for Low-Resource Myanmar ASR",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-bg text-slate-800 antialiased overflow-hidden">
        {children}
      </body>
    </html>
  );
}

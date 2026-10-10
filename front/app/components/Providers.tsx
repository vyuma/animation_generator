"use client";

import { ReactNode } from "react";
import { TextProvider } from "@/app/context/TextContext";
import { VideoGenerationProvider } from "@/app/context/VideoGenerationContext";

export default function Providers({ children }: { children: ReactNode }) {
  return (
    <TextProvider>
      <VideoGenerationProvider>{children}</VideoGenerationProvider>
    </TextProvider>
  );
}

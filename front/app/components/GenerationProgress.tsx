"use client";

import React from "react";
import { motion } from "framer-motion";
import { JobStatus, JobStep } from "@/app/type/apiTypes";

interface GenerationProgressProps {
  status: JobStatus | null;
  currentStep: JobStep | null;
  progress: number;
  error: string | null;
}

// ステップのラベルマッピング
const stepLabels: Record<JobStep, string> = {
  initializing: "初期化中...",
  planning: "プランニング中...",
  generating_script: "スクリプト生成中...",
  linting: "コード検証中...",
  security_check: "セキュリティチェック中...",
  preflight: "プリフライトチェック中...",
  executing: "動画生成中...",
  refining: "最適化中...",
  finalizing: "完了処理中...",
};

// ステップの順序
const stepOrder: JobStep[] = [
  "initializing",
  "planning",
  "generating_script",
  "linting",
  "security_check",
  "preflight",
  "executing",
  "refining",
  "finalizing",
];

const GenerationProgress: React.FC<GenerationProgressProps> = ({
  status,
  currentStep,
  progress,
  error,
}) => {
  if (!status || status === "completed") {
    return null;
  }

  const currentStepIndex = currentStep
    ? stepOrder.indexOf(currentStep)
    : 0;

  return (
    <div className="w-full p-4 bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-cyan-400/40 dark:border-cyan-500/30 transition-colors">
      {/* ステータスヘッダー */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {status === "running" && (
            <motion.div
              className="w-3 h-3 bg-blue-500 rounded-full"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
          )}
          {status === "pending" && (
            <div className="w-3 h-3 bg-yellow-500 rounded-full" />
          )}
          {status === "failed" && (
            <div className="w-3 h-3 bg-red-500 rounded-full" />
          )}
          <span className="font-medium text-slate-700 dark:text-slate-200">
            {status === "running" && "生成中"}
            {status === "pending" && "待機中"}
            {status === "failed" && "エラー"}
          </span>
        </div>
        <span className="text-sm text-cyan-600 dark:text-cyan-400">
          {Math.round(progress)}%
        </span>
      </div>

      {/* プログレスバー */}
      <div className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden mb-4">
        <motion.div
          className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full shadow-lg shadow-cyan-500/50"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        />
      </div>

      {/* 現在のステップ */}
      {currentStep && status !== "failed" && (
        <div className="flex items-center gap-2 mb-4">
          <motion.div
            className="flex gap-1"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            <span className="w-1.5 h-1.5 bg-cyan-500 rounded-full" />
            <span className="w-1.5 h-1.5 bg-cyan-500 rounded-full" />
            <span className="w-1.5 h-1.5 bg-cyan-500 rounded-full" />
          </motion.div>
          <span className="text-sm text-slate-600 dark:text-slate-300">
            {stepLabels[currentStep]}
          </span>
        </div>
      )}

      {/* ステップインジケーター */}
      <div className="flex justify-between items-center">
        {stepOrder.map((step, index) => (
          <div key={step} className="flex flex-col items-center">
            <motion.div
              className={`w-2 h-2 rounded-full ${
                index < currentStepIndex
                  ? "bg-cyan-400"
                  : index === currentStepIndex
                  ? "bg-cyan-500 shadow-lg shadow-cyan-500/50"
                  : "bg-slate-300 dark:bg-slate-600"
              }`}
              animate={
                index === currentStepIndex
                  ? { scale: [1, 1.3, 1] }
                  : {}
              }
              transition={{ duration: 0.8, repeat: Infinity }}
            />
          </div>
        ))}
      </div>

      {/* エラーメッセージ */}
      {error && (
        <motion.div
          className="mt-4 p-3 bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </motion.div>
      )}
    </div>
  );
};

export default GenerationProgress;

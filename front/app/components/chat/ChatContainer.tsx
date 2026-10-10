"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import ChatMessage from "./ChatMessage";
import PlanPreview from "./PlanPreview";
import { useVideoGeneration } from "@/app/context/VideoGenerationContext";

type GenerationMode = "full" | "planning";

export default function ChatContainer() {
  const [input, setInput] = useState("");
  const [generationMode, setGenerationMode] = useState<GenerationMode>("full");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const {
    getCurrentSession,
    sendMessage,
    isLoading,
    isLoadingSession,
    confirmPlan,
    editPlan,
    updatePlan,
    cancelPlan,
  } = useVideoGeneration();

  const currentSession = getCurrentSession();
  const messages = currentSession?.messages || [];
  const isNewSession = !currentSession || messages.length === 0;
  const isPlanReview = currentSession?.phase === "plan_review";
  const isPlanning = currentSession?.phase === "planning";
  const isGenerating = currentSession?.phase === "generating";

  // 新しいメッセージが追加されたら自動スクロール
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || isPlanning || isGenerating) return;

    const message = input.trim();
    setInput("");
    // 新規セッションの場合のみモード選択を適用
    await sendMessage(message, isNewSession && generationMode === "planning");
  };

  const handleConfirmPlan = async () => {
    await confirmPlan();
  };

  const handleEditPlan = () => {
    editPlan();
  };

  const handleUpdatePlan = (content: string) => {
    updatePlan(content);
  };

  const handleCancelPlan = () => {
    cancelPlan();
  };

  return (
    <div className="flex flex-col h-full">
      {/* メッセージエリア */}
      <div className="flex-1 overflow-y-auto p-2 sm:p-4 chat-scrollbar">
        <div className="max-w-4xl mx-auto h-full">
          {isLoadingSession ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 dark:text-slate-400">
              <motion.div
                className="w-10 h-10 sm:w-12 sm:h-12 border-4 border-slate-300 dark:border-slate-600 border-t-cyan-500 dark:border-t-cyan-400 rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              <p className="text-base sm:text-lg mt-4">セッションを読み込み中...</p>
            </div>
          ) : messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 dark:text-slate-400 px-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="w-12 h-12 sm:w-16 sm:h-16 mb-4 text-cyan-400/50 dark:text-cyan-500/30"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              >
                <path d="M15 10l4.553-2.276A1 1 0 0 1 21 8.618v6.764a1 1 0 0 1-1.447.894L15 14v-4zM3 8a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8z" />
              </svg>
              <p className="text-base sm:text-lg mb-2 text-center">アニメーションを作成しましょう</p>
              <p className="text-xs sm:text-sm text-slate-400 dark:text-slate-500 text-center">
                作りたいアニメーションの説明を入力してください
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {/* プランニング中のローディング表示 */}
              {isPlanning && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start mb-4"
                >
                  <div className="max-w-[90%] sm:max-w-[80%] px-3 sm:px-4 py-3 sm:py-4 bg-teal-50 dark:bg-teal-900/20 border border-teal-400/40 dark:border-teal-500/30 rounded-2xl rounded-bl-sm shadow-md">
                    <div className="flex items-center gap-2 sm:gap-3">
                      {/* 動的なアイコンアニメーション */}
                      <div className="relative">
                        <motion.div
                          className="w-6 h-6 sm:w-8 sm:h-8 border-2 border-teal-400/30 dark:border-teal-500/30 border-t-teal-500 dark:border-t-teal-400 rounded-full"
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        />
                        <motion.div
                          className="absolute inset-0 flex items-center justify-center"
                          animate={{ scale: [1, 1.1, 1] }}
                          transition={{ duration: 1.5, repeat: Infinity }}
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="w-3 h-3 sm:w-4 sm:h-4 text-teal-500 dark:text-teal-400"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                          >
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                          </svg>
                        </motion.div>
                      </div>
                      <div>
                        <div className="flex items-center gap-1">
                          <span className="text-xs sm:text-sm font-medium text-teal-600 dark:text-teal-400">
                            プランを作成中
                          </span>
                          {/* ドットアニメーション */}
                          <motion.span
                            className="flex gap-0.5"
                            initial={{ opacity: 1 }}
                          >
                            {[0, 1, 2].map((i) => (
                              <motion.span
                                key={i}
                                className="w-1 h-1 bg-teal-500 dark:bg-teal-400 rounded-full"
                                animate={{ opacity: [0.3, 1, 0.3] }}
                                transition={{
                                  duration: 1,
                                  repeat: Infinity,
                                  delay: i * 0.2,
                                }}
                              />
                            ))}
                          </motion.span>
                        </div>
                        <p className="text-[10px] sm:text-xs text-teal-500/70 dark:text-teal-400/70 mt-0.5">
                          AIがアニメーションの構成を考えています
                        </p>
                      </div>
                    </div>
                    {/* プログレスバー風アニメーション */}
                    <div className="mt-2 sm:mt-3 w-full h-1 sm:h-1.5 bg-teal-200/50 dark:bg-teal-800/30 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-teal-400 via-teal-500 to-teal-400"
                        animate={{
                          x: ["-100%", "100%"],
                        }}
                        transition={{
                          duration: 1.5,
                          repeat: Infinity,
                          ease: "linear",
                        }}
                        style={{ width: "50%" }}
                      />
                    </div>
                  </div>
                </motion.div>
              )}
              {/* プランプレビュー */}
              {isPlanReview && currentSession?.plan && (
                <PlanPreview
                  plan={currentSession.plan}
                  onConfirm={handleConfirmPlan}
                  onEdit={handleEditPlan}
                  onUpdate={handleUpdatePlan}
                  onCancel={handleCancelPlan}
                />
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* 入力エリア */}
      {!isPlanReview && (
        <div className="border-t border-cyan-400/30 dark:border-cyan-500/20 p-2 sm:p-4 pb-[calc(0.5rem+env(safe-area-inset-bottom))] sm:pb-[calc(1rem+env(safe-area-inset-bottom))] bg-white/80 dark:bg-slate-800/50 transition-colors">
          <div className="max-w-4xl mx-auto">
            {/* モード選択（新規セッションのみ表示） */}
            {isNewSession && (
              <div className="flex gap-2 mb-2 sm:mb-3">
                <button
                  type="button"
                  onClick={() => setGenerationMode("full")}
                  className={`flex-1 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                    generationMode === "full"
                      ? "bg-cyan-600 text-white shadow-md"
                      : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600"
                  }`}
                >
                  <span className="flex items-center justify-center gap-1 sm:gap-2">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-3 h-3 sm:w-4 sm:h-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    <span className="hidden xs:inline">フル</span>生成
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => setGenerationMode("planning")}
                  className={`flex-1 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                    generationMode === "planning"
                      ? "bg-purple-600 text-white shadow-md"
                      : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600"
                  }`}
                >
                  <span className="flex items-center justify-center gap-1 sm:gap-2">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-3 h-3 sm:w-4 sm:h-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                      <line x1="16" y1="13" x2="8" y2="13" />
                      <line x1="16" y1="17" x2="8" y2="17" />
                    </svg>
                    プラン
                  </span>
                </button>
              </div>
            )}
            <form onSubmit={handleSubmit} className="flex gap-2 sm:gap-3">
              <div className="flex-1 relative">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                  placeholder={
                    isNewSession && generationMode === "planning"
                      ? "アニメーションの説明を入力..."
                      : "アニメーションの説明を入力..."
                  }
                  disabled={isLoading || isLoadingSession || isPlanning || isGenerating}
                  className="w-full px-3 sm:px-4 py-2 sm:py-3 bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-white rounded-xl border border-slate-300 dark:border-slate-600 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none resize-none transition-all placeholder-slate-400 disabled:opacity-50 text-sm sm:text-base"
                  rows={2}
                />
              </div>
              <button
                type="submit"
                disabled={!input.trim() || isLoading || isLoadingSession || isPlanning || isGenerating}
                className={`px-4 sm:px-6 py-2 sm:py-3 rounded-xl flex items-center justify-center transition-all ${
                  input.trim() && !isLoading && !isLoadingSession && !isPlanning && !isGenerating
                    ? generationMode === "planning" && isNewSession
                      ? "bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-500/30"
                      : "bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-500/30"
                    : "bg-slate-300 dark:bg-slate-600 text-slate-400 cursor-not-allowed"
                }`}
              >
                {isLoading || isPlanning || isGenerating ? (
                  <motion.div
                    className="w-4 h-4 sm:w-5 sm:h-5 border-2 border-white/30 border-t-white rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                ) : (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4 sm:w-5 sm:h-5"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M3 20V4l19 8-19 8zm2-3l11.85-5L5 7v3.5l6 1.5-6 1.5V17z" />
                  </svg>
                )}
              </button>
            </form>
            <p className="hidden sm:block text-xs text-slate-400 dark:text-slate-500 mt-2">
              {isNewSession && generationMode === "planning"
                ? "Enter で改行、Cmd + Enter でプラン作成"
                : "Enter で改行、Cmd + Enter で送信"}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

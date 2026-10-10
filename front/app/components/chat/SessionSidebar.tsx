"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GenerationSession } from "@/app/type/apiTypes";

interface SessionSidebarProps {
  sessions: GenerationSession[];
  currentSessionId: number | null;
  onSelectSession: (sessionId: number) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: number) => void;
  isLoadingSession?: boolean;
  isOpen?: boolean;
  onClose?: () => void;
}

export default function SessionSidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  isLoadingSession = false,
  isOpen = true,
  onClose,
}: SessionSidebarProps) {
  // 一時セッション（generationId: null）を除外した既存セッション
  const existingSessions = sessions.filter((s) => s.generationId !== null);
  // 一時セッション
  const tempSession = sessions.find((s) => s.generationId === null);

  return (
    <>
      {/* モバイル用オーバーレイ背景 */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 top-14 md:top-16 bg-black/50 z-40 md:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* サイドバー */}
      <motion.div
        initial={false}
        animate={{
          width: isOpen ? 256 : 0,
          opacity: isOpen ? 1 : 0,
        }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
        className={`
          h-[calc(100dvh-3.5rem)] md:h-full bg-slate-100 dark:bg-slate-800 border-r border-cyan-400/30 dark:border-cyan-500/20
          flex flex-col transition-colors overflow-hidden
          fixed md:relative z-50 md:z-auto top-14 md:top-0 left-0
          ${isOpen ? "w-64" : "w-0"}
        `}
      >
        <div className="w-64 flex-1 flex flex-col overflow-hidden">
          {/* ヘッダー */}
          <div className="p-3 md:p-4 border-b border-cyan-400/30 dark:border-cyan-500/20 flex items-center gap-2">
            <button
              onClick={onNewSession}
              className="flex-1 px-3 md:px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors flex items-center justify-center gap-2 text-sm md:text-base"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="w-4 h-4 md:w-5 md:h-5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M12 5v14M5 12h14" />
              </svg>
              新規作成
            </button>
            {/* モバイル用閉じるボタン */}
            <button
              onClick={onClose}
              className="md:hidden p-2 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
              aria-label="サイドバーを閉じる"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="w-5 h-5 text-slate-600 dark:text-slate-300"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* セッションリスト */}
          <div className="flex-1 overflow-y-auto p-2 scrollbar-thin">
            {existingSessions.length === 0 && !tempSession ? (
              <p className="text-slate-500 dark:text-slate-400 text-sm text-center py-4">
                セッションがありません
              </p>
            ) : (
              <div className="space-y-2">
                {/* 一時セッション（新規作成中）*/}
                {tempSession && (
                  <motion.div
                    key="temp-session"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`group relative p-3 rounded-lg cursor-pointer transition-all ${
                      currentSessionId === null
                        ? "bg-cyan-500/20 dark:bg-cyan-600/20 border border-cyan-500/50"
                        : "bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-200 dark:hover:bg-slate-700 border border-transparent"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-slate-700 dark:text-slate-200 truncate">
                          {tempSession.title || "新規セッション"}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">作成中...</p>
                      </div>
                      {/* ステータスインジケーター */}
                      <div className="ml-2">
                        {(tempSession.status === "running" ||
                          tempSession.status === "pending") && (
                          <motion.div
                            className="w-2 h-2 bg-cyan-500 rounded-full"
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity }}
                          />
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* 既存セッション */}
                {existingSessions.map((session) => (
                  <motion.div
                    key={session.generationId}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`group relative p-3 rounded-lg cursor-pointer transition-all ${
                      currentSessionId === session.generationId
                        ? "bg-cyan-500/20 dark:bg-cyan-600/20 border border-cyan-500/50"
                        : "bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-200 dark:hover:bg-slate-700 border border-transparent"
                    }`}
                    onClick={() =>
                      session.generationId && onSelectSession(session.generationId)
                    }
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-slate-700 dark:text-slate-200 truncate">
                          {session.title || `セッション #${session.generationId}`}
                        </p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                          {session.createdAt.toLocaleDateString()}
                        </p>
                      </div>
                      {/* ステータスインジケーター */}
                      <div className="ml-2">
                        {session.status === "running" && (
                          <motion.div
                            className="w-2 h-2 bg-cyan-500 rounded-full"
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity }}
                          />
                        )}
                        {session.status === "completed" && (
                          <div className="w-2 h-2 bg-green-500 rounded-full" />
                        )}
                        {session.status === "failed" && (
                          <div className="w-2 h-2 bg-red-500 rounded-full" />
                        )}
                        {isLoadingSession &&
                          currentSessionId === session.generationId && (
                            <motion.div
                              className="w-2 h-2 bg-yellow-500 rounded-full"
                              animate={{ opacity: [1, 0.5, 1] }}
                              transition={{ duration: 1, repeat: Infinity }}
                            />
                          )}
                      </div>
                    </div>

                    {/* 削除ボタン */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (session.generationId) {
                          onDeleteSession(session.generationId);
                        }
                      }}
                      className="absolute top-2 right-2 p-1 opacity-0 group-hover:opacity-100 hover:bg-red-500/20 rounded transition-all"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="w-4 h-4 text-red-400"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M18 6L6 18M6 6l12 12" />
                      </svg>
                    </button>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </>
  );
}

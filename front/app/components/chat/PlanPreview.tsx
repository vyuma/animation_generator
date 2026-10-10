"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { PlanInfo } from "@/app/type/apiTypes";

interface PlanPreviewProps {
  plan: PlanInfo;
  onConfirm: () => void;
  onEdit: () => void;
  onUpdate: (content: string) => void;
  onCancel: () => void;
}

type EditTab = "edit" | "preview";

// Markdownコンポーネントの共通定義
const markdownComponents = {
  h1: ({ children }: { children?: React.ReactNode }) => (
    <h1 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2 mt-3 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100 mb-2 mt-3">
      {children}
    </h2>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-1 mt-2">
      {children}
    </h3>
  ),
  p: ({ children }: { children?: React.ReactNode }) => (
    <p className="text-sm text-slate-700 dark:text-slate-300 mb-2 leading-relaxed">
      {children}
    </p>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <ul className="text-sm text-slate-700 dark:text-slate-300 list-disc list-inside mb-2 space-y-1">
      {children}
    </ul>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <ol className="text-sm text-slate-700 dark:text-slate-300 list-decimal list-inside mb-2 space-y-1">
      {children}
    </ol>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <li className="text-sm text-slate-700 dark:text-slate-300">{children}</li>
  ),
  code: ({ children }: { children?: React.ReactNode }) => (
    <code className="bg-slate-200 dark:bg-slate-700 px-1 py-0.5 rounded text-xs font-mono text-teal-600 dark:text-teal-400">
      {children}
    </code>
  ),
  pre: ({ children }: { children?: React.ReactNode }) => (
    <pre className="bg-slate-200 dark:bg-slate-700 p-3 rounded-lg overflow-x-auto mb-2 code-scrollbar">
      {children}
    </pre>
  ),
  strong: ({ children }: { children?: React.ReactNode }) => (
    <strong className="font-semibold text-slate-800 dark:text-slate-100">
      {children}
    </strong>
  ),
  em: ({ children }: { children?: React.ReactNode }) => (
    <em className="italic text-slate-600 dark:text-slate-400">{children}</em>
  ),
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <blockquote className="border-l-4 border-teal-400 dark:border-teal-500 pl-3 italic text-slate-600 dark:text-slate-400 my-2">
      {children}
    </blockquote>
  ),
};

export default function PlanPreview({
  plan,
  onConfirm,
  onEdit,
  onUpdate,
  onCancel,
}: PlanPreviewProps) {
  const [editedContent, setEditedContent] = useState(plan.content);
  const [isConfirming, setIsConfirming] = useState(false);
  const [editTab, setEditTab] = useState<EditTab>("edit");

  useEffect(() => {
    setEditedContent(plan.content);
  }, [plan.content]);

  const handleConfirm = async () => {
    setIsConfirming(true);
    await onConfirm();
  };

  const handleSaveEdit = () => {
    onUpdate(editedContent);
  };

  const handleCancelEdit = () => {
    setEditedContent(plan.content);
    onUpdate(plan.content); // Reset to original and exit edit mode
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex justify-start mb-4"
    >
      <div className="max-w-[90%] w-full p-4 bg-teal-50 dark:bg-teal-900/20 border border-teal-400/40 dark:border-teal-500/30 rounded-2xl rounded-bl-sm shadow-md">
        <div className="flex items-center gap-2 mb-3">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-5 h-5 text-teal-600 dark:text-teal-400"
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
          <span className="text-sm font-medium text-teal-600 dark:text-teal-400">
            プラン
          </span>
        </div>

        {plan.isEditing ? (
          <div className="space-y-3">
            {/* タブ切り替え */}
            <div className="flex border-b border-teal-300 dark:border-teal-600">
              <button
                type="button"
                onClick={() => setEditTab("edit")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  editTab === "edit"
                    ? "text-teal-600 dark:text-teal-400 border-b-2 border-teal-600 dark:border-teal-400"
                    : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
                }`}
              >
                <span className="flex items-center gap-1.5">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                  編集
                </span>
              </button>
              <button
                type="button"
                onClick={() => setEditTab("preview")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  editTab === "preview"
                    ? "text-teal-600 dark:text-teal-400 border-b-2 border-teal-600 dark:border-teal-400"
                    : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
                }`}
              >
                <span className="flex items-center gap-1.5">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                  プレビュー
                </span>
              </button>
            </div>

            {/* 編集/プレビューエリア */}
            {editTab === "edit" ? (
              <textarea
                value={editedContent}
                onChange={(e) => setEditedContent(e.target.value)}
                className="w-full h-64 p-3 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-lg border border-teal-300 dark:border-teal-600 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 outline-none resize-none text-sm font-mono scrollbar-thin"
                placeholder="Markdown形式で入力..."
              />
            ) : (
              <div className="w-full h-64 p-4 bg-white dark:bg-slate-800 rounded-lg border border-teal-300 dark:border-teal-600 overflow-y-auto scrollbar-thin">
                {editedContent ? (
                  <ReactMarkdown components={markdownComponents}>
                    {editedContent}
                  </ReactMarkdown>
                ) : (
                  <p className="text-sm text-slate-400 dark:text-slate-500 italic">
                    プレビューする内容がありません
                  </p>
                )}
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={handleSaveEdit}
                className="px-4 py-2 bg-teal-600 hover:bg-teal-500 text-white text-sm font-medium rounded-lg transition-colors"
              >
                保存
              </button>
              <button
                onClick={handleCancelEdit}
                className="px-4 py-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 text-sm font-medium rounded-lg transition-colors"
              >
                キャンセル
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4 mb-4 max-h-96 overflow-y-auto scrollbar-thin prose prose-sm dark:prose-invert prose-slate max-w-none">
              <ReactMarkdown components={markdownComponents}>
                {plan.content}
              </ReactMarkdown>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleConfirm}
                disabled={isConfirming}
                className={`flex-1 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  isConfirming
                    ? "bg-emerald-400 cursor-not-allowed"
                    : "bg-emerald-600 hover:bg-emerald-500"
                } text-white`}
              >
                {isConfirming ? (
                  <span className="flex items-center justify-center gap-2">
                    <motion.span
                      className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full inline-block"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    />
                    生成中...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-4 h-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    このプランで生成
                  </span>
                )}
              </button>
              <button
                onClick={onEdit}
                disabled={isConfirming}
                className="px-4 py-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span className="flex items-center gap-2">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                  編集
                </span>
              </button>
              <button
                onClick={onCancel}
                disabled={isConfirming}
                className="px-4 py-2 bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-900/50 text-red-600 dark:text-red-400 text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span className="flex items-center gap-2">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                  キャンセル
                </span>
              </button>
            </div>
          </>
        )}
      </div>
    </motion.div>
  );
}

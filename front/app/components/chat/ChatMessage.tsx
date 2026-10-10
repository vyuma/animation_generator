"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import ReactMarkdown, { Components } from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { ChatMessage as ChatMessageType } from "@/app/type/apiTypes";
import { registerVideoToRag } from "@/app/lib/animationApi";

interface ChatMessageProps {
  message: ChatMessageType;
}

const stepLabels: Record<string, string> = {
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

// ユーザーメッセージ用のMarkdownコンポーネント（白テキスト、高コントラスト）
const userMarkdownComponents: Components = {
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-white mb-2 mt-2 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-semibold text-white mb-1.5 mt-2">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-cyan-50 mb-1 mt-1.5">
      {children}
    </h3>
  ),
  p: ({ children }) => (
    <p className="text-sm text-white mb-2 last:mb-0 leading-relaxed">
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul className="text-sm text-white list-disc list-inside mb-2 space-y-0.5">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="text-sm text-white list-decimal list-inside mb-2 space-y-0.5">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="text-sm text-white">{children}</li>
  ),
  code: ({ children }) => (
    <code className="bg-cyan-900/60 px-1 py-0.5 rounded text-xs font-mono text-cyan-50">
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="bg-cyan-900/60 p-2 rounded-lg overflow-x-auto mb-2 text-xs code-scrollbar-user">
      {children}
    </pre>
  ),
  strong: ({ children }) => (
    <strong className="font-bold text-white">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-cyan-50">{children}</em>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-cyan-200 pl-3 my-2 text-cyan-50 italic">
      {children}
    </blockquote>
  ),
};

// システム/アシスタントメッセージ用のMarkdownコンポーネント（ダークモード対応）
const systemMarkdownComponents: Components = {
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-2 mt-2 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100 mb-1.5 mt-2">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-1 mt-1.5">
      {children}
    </h3>
  ),
  p: ({ children }) => (
    <p className="text-sm text-slate-700 dark:text-slate-300 mb-2 last:mb-0 leading-relaxed">
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul className="text-sm text-slate-700 dark:text-slate-300 list-disc list-inside mb-2 space-y-0.5">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="text-sm text-slate-700 dark:text-slate-300 list-decimal list-inside mb-2 space-y-0.5">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="text-sm text-slate-700 dark:text-slate-300">{children}</li>
  ),
  code: ({ children }) => (
    <code className="bg-slate-300 dark:bg-slate-600 px-1 py-0.5 rounded text-xs font-mono text-slate-800 dark:text-slate-200">
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="bg-slate-300 dark:bg-slate-600 p-2 rounded-lg overflow-x-auto mb-2 text-xs code-scrollbar">
      {children}
    </pre>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-800 dark:text-slate-100">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-slate-600 dark:text-slate-400">{children}</em>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-slate-400 dark:border-slate-500 pl-3 my-2 text-slate-600 dark:text-slate-400 italic">
      {children}
    </blockquote>
  ),
};

// エラーメッセージ用のMarkdownコンポーネント
const errorMarkdownComponents: Components = {
  p: ({ children }) => (
    <p className="text-sm text-red-600 dark:text-red-400 mb-2 last:mb-0 leading-relaxed">
      {children}
    </p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-red-700 dark:text-red-300">{children}</strong>
  ),
  code: ({ children }) => (
    <code className="bg-red-200 dark:bg-red-800/50 px-1 py-0.5 rounded text-xs font-mono">
      {children}
    </code>
  ),
};

// プランメッセージ用のMarkdownコンポーネント
const planMarkdownComponents: Components = {
  h1: ({ children }) => (
    <h1 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2 mt-3 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-base font-semibold text-slate-800 dark:text-slate-100 mb-2 mt-3">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-1 mt-2">
      {children}
    </h3>
  ),
  p: ({ children }) => (
    <p className="text-sm text-slate-700 dark:text-slate-300 mb-2 leading-relaxed">
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul className="text-sm text-slate-700 dark:text-slate-300 list-disc list-inside mb-2 space-y-1">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="text-sm text-slate-700 dark:text-slate-300 list-decimal list-inside mb-2 space-y-1">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="text-sm text-slate-700 dark:text-slate-300">{children}</li>
  ),
  code: ({ children }) => (
    <code className="bg-slate-200 dark:bg-slate-700 px-1 py-0.5 rounded text-xs font-mono text-teal-600 dark:text-teal-400">
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre className="bg-slate-200 dark:bg-slate-700 p-3 rounded-lg overflow-x-auto mb-2 code-scrollbar">
      {children}
    </pre>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-800 dark:text-slate-100">
      {children}
    </strong>
  ),
};

// いいね済みvideoIdをlocalStorageに保存するキー
const LIKED_VIDEOS_KEY = "physiquest_liked_videos";

// いいね済みvideoIdリストを取得
const getLikedVideos = (): string[] => {
  if (typeof window === "undefined") return [];
  try {
    const stored = localStorage.getItem(LIKED_VIDEOS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
};

// videoIdをいいね済みリストに追加
const addLikedVideo = (videoId: string): void => {
  if (typeof window === "undefined") return;
  try {
    const liked = getLikedVideos();
    if (!liked.includes(videoId)) {
      liked.push(videoId);
      localStorage.setItem(LIKED_VIDEOS_KEY, JSON.stringify(liked));
    }
  } catch (e) {
    console.error("Failed to save liked video:", e);
  }
};

// videoIdがいいね済みかチェック
const isVideoLiked = (videoId: string): boolean => {
  return getLikedVideos().includes(videoId);
};

// 動画メッセージコンポーネント（いいねボタン付き）
function VideoMessage({ message }: { message: ChatMessageType }) {
  const [isLiked, setIsLiked] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 初期化時にlocalStorageからいいね状態を読み込む
  useEffect(() => {
    if (message.videoId) {
      setIsLiked(isVideoLiked(message.videoId));
    }
  }, [message.videoId]);

  const handleLike = async () => {
    if (!message.videoId || isLiked || isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      await registerVideoToRag(message.videoId);
      // 成功したらlocalStorageに保存
      addLikedVideo(message.videoId);
      setIsLiked(true);
    } catch (err) {
      console.error("RAG registration failed:", err);
      setError(err instanceof Error ? err.message : "登録に失敗しました");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex justify-start mb-3 sm:mb-4"
    >
      <div className="max-w-[95%] sm:max-w-[90%] p-2 sm:p-4 bg-slate-100 dark:bg-slate-800 border border-cyan-400/40 dark:border-cyan-500/30 rounded-2xl rounded-bl-sm shadow-md">
        <p className="text-xs sm:text-sm text-cyan-600 dark:text-cyan-400 mb-2 sm:mb-3">動画が生成されました</p>
        <video
          controls
          className="w-full rounded-lg sm:rounded-xl border border-cyan-400/30 dark:border-cyan-500/20"
          autoPlay
          muted
          playsInline
        >
          <source src={message.videoUrl} type="video/mp4" />
        </video>
        <div className="flex flex-wrap items-center gap-2 mt-2 sm:mt-3">
          <a
            href={message.videoUrl}
            download="animation.mp4"
            className="px-2 sm:px-3 py-1 sm:py-1.5 text-[10px] sm:text-xs bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors"
          >
            ダウンロード
          </a>
          {message.videoId && (
            <button
              onClick={handleLike}
              disabled={isLiked || isLoading}
              className={`flex items-center gap-1 sm:gap-1.5 px-2 sm:px-3 py-1 sm:py-1.5 text-[10px] sm:text-xs rounded-lg transition-colors ${
                isLiked
                  ? "bg-pink-500 text-white cursor-default"
                  : isLoading
                  ? "bg-slate-400 text-white cursor-wait"
                  : "bg-slate-300 dark:bg-slate-600 hover:bg-pink-400 hover:text-white text-slate-700 dark:text-slate-200"
              }`}
              title={isLiked ? "テンプレートに登録済み" : "テンプレートとして登録"}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className={`w-3 h-3 sm:w-4 sm:h-4 ${isLiked ? "fill-current" : ""}`}
                viewBox="0 0 24 24"
                fill={isLiked ? "currentColor" : "none"}
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
              </svg>
              <span className="hidden xs:inline">{isLoading ? "登録中..." : isLiked ? "登録済み" : "いいね"}</span>
              <span className="xs:hidden">{isLoading ? "..." : isLiked ? "済" : ""}</span>
            </button>
          )}
        </div>
        {error && (
          <p className="text-[10px] sm:text-xs text-red-500 mt-1">{error}</p>
        )}
        <span className="text-[10px] sm:text-xs text-slate-500 dark:text-slate-400 mt-2 block">
          {message.timestamp.toLocaleTimeString()}
        </span>
      </div>
    </motion.div>
  );
}

export default function ChatMessage({ message }: ChatMessageProps) {
  // ユーザーメッセージ
  if (message.type === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-end mb-3 sm:mb-4"
      >
        <div className="max-w-[85%] sm:max-w-[80%] px-3 sm:px-4 py-2 sm:py-3 bg-cyan-700 dark:bg-cyan-800 text-white rounded-2xl rounded-br-sm shadow-md">
          <div className="prose prose-sm prose-invert max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={userMarkdownComponents}
            >
              {message.content}
            </ReactMarkdown>
          </div>
          <span className="text-[10px] sm:text-xs text-cyan-300 mt-1 block">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </motion.div>
    );
  }

  // システムメッセージ
  if (message.type === "system") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-start mb-3 sm:mb-4"
      >
        <div className="max-w-[85%] sm:max-w-[80%] px-3 sm:px-4 py-2 sm:py-3 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-2xl rounded-bl-sm shadow-md">
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={systemMarkdownComponents}
            >
              {message.content}
            </ReactMarkdown>
          </div>
          <span className="text-[10px] sm:text-xs text-slate-500 dark:text-slate-400 mt-1 block">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </motion.div>
    );
  }

  // 進捗メッセージ
  if (message.type === "progress") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-start mb-3 sm:mb-4"
      >
        <div className="max-w-[90%] sm:max-w-[80%] px-3 sm:px-4 py-2 sm:py-3 bg-slate-100 dark:bg-slate-800 border border-cyan-400/40 dark:border-cyan-500/30 rounded-2xl rounded-bl-sm shadow-md">
          <div className="flex items-center gap-2 mb-2">
            <motion.div
              className="w-2 h-2 bg-cyan-500 rounded-full"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
            <span className="text-xs sm:text-sm text-cyan-600 dark:text-cyan-400">
              {message.currentStep ? stepLabels[message.currentStep] || message.currentStep : "処理中..."}
            </span>
          </div>
          <div className="w-full h-1.5 sm:h-2 bg-slate-300 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${message.progress || 0}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <span className="text-[10px] sm:text-xs text-slate-500 dark:text-slate-400 mt-2 block">
            {Math.round(message.progress || 0)}%
          </span>
        </div>
      </motion.div>
    );
  }

  // エラーメッセージ
  if (message.type === "error") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-start mb-3 sm:mb-4"
      >
        <div className="max-w-[90%] sm:max-w-[80%] px-3 sm:px-4 py-2 sm:py-3 bg-red-100 dark:bg-red-900/30 border border-red-400/50 dark:border-red-500/50 text-red-600 dark:text-red-400 rounded-2xl rounded-bl-sm shadow-md">
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={errorMarkdownComponents}
            >
              {message.content}
            </ReactMarkdown>
          </div>
          <span className="text-[10px] sm:text-xs text-red-500/70 mt-1 block">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </motion.div>
    );
  }

  // プランメッセージ
  if (message.type === "plan" && message.planContent) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-start mb-3 sm:mb-4"
      >
        <div className="max-w-[95%] sm:max-w-[90%] w-full p-2 sm:p-4 bg-teal-50 dark:bg-teal-900/20 border border-teal-400/40 dark:border-teal-500/30 rounded-2xl rounded-bl-sm shadow-md">
          <div className="flex items-center gap-2 mb-2 sm:mb-3">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-4 h-4 sm:w-5 sm:h-5 text-teal-600 dark:text-teal-400"
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
            <span className="text-xs sm:text-sm font-medium text-teal-600 dark:text-teal-400">
              {message.content}
            </span>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-lg p-2 sm:p-4 max-h-48 sm:max-h-64 overflow-y-auto scrollbar-thin">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={planMarkdownComponents}
            >
              {message.planContent}
            </ReactMarkdown>
          </div>
          <span className="text-[10px] sm:text-xs text-slate-500 dark:text-slate-400 mt-2 block">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </motion.div>
    );
  }

  // 動画メッセージ
  if (message.type === "video" && message.videoUrl) {
    return (
      <VideoMessage message={message} />
    );
  }

  return null;
}

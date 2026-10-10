"use client";

import React, { useState } from "react";
import LoadingDots from "@/app/components/loadingAnimation";

interface EditFormProps {
  onSubmit: (enhancePrompt: string) => void;
  loading: boolean;
  disabled?: boolean;
}

const EditForm: React.FC<EditFormProps> = ({
  onSubmit,
  loading,
  disabled = false,
}) => {
  const [enhancePrompt, setEnhancePrompt] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!enhancePrompt.trim() || loading || disabled) return;
    onSubmit(enhancePrompt);
    setEnhancePrompt("");
  };

  return (
    <div className="w-full p-4 bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-cyan-400/40 dark:border-cyan-500/30 transition-colors">
      <h3 className="text-sm font-medium text-cyan-600 dark:text-cyan-400 mb-3">
        動画を編集
      </h3>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <textarea
          className="w-full h-24 px-4 py-3 outline-none resize-none rounded-xl bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 placeholder-slate-400 text-sm border border-slate-300 dark:border-slate-600 focus:border-cyan-500/50 transition-all"
          placeholder="編集の指示を入力してください（例：色を赤に変更、速度を2倍に）"
          value={enhancePrompt}
          onChange={(e) => setEnhancePrompt(e.target.value)}
          disabled={loading || disabled}
        />
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!enhancePrompt.trim() || loading || disabled}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 text-sm font-medium transition-all ${
              enhancePrompt.trim() && !loading && !disabled
                ? "bg-cyan-500 hover:bg-cyan-400 text-white shadow-lg shadow-cyan-500/30"
                : "bg-slate-300 dark:bg-slate-600 text-slate-400 cursor-not-allowed"
            }`}
          >
            {loading ? (
              <>
                <LoadingDots />
                <span>編集中...</span>
              </>
            ) : (
              <>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-4 h-4"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                </svg>
                <span>編集を適用</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default EditForm;

"use client";

import React from "react";
import ToggleSwitch from "./ToggleSwitch";

interface HeadersProps {
  onMenuClick?: () => void;
  isSidebarOpen?: boolean;
}

const Headers = ({ onMenuClick, isSidebarOpen }: HeadersProps) => {
  return (
    <div className="w-full bg-white dark:bg-slate-800 h-14 md:h-16 flex justify-between items-center px-3 md:px-6 pt-[env(safe-area-inset-top)] shadow-lg border-b border-cyan-400/40 dark:border-cyan-500/30 transition-colors flex-shrink-0">
      <div className="flex items-center gap-2 md:gap-4">
        {/* ハンバーガーメニュー / サイドバートグル */}
        <button
          onClick={onMenuClick}
          className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
          aria-label={isSidebarOpen ? "サイドバーを閉じる" : "サイドバーを開く"}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-5 h-5 md:w-6 md:h-6 text-slate-600 dark:text-slate-300"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            {isSidebarOpen ? (
              // サイドバー閉じるアイコン
              <path d="M11 19l-7-7 7-7M18 5v14" />
            ) : (
              // ハンバーガーメニューアイコン
              <>
                <path d="M3 12h18M3 6h18M3 18h18" />
              </>
            )}
          </svg>
        </button>

        <h1 className="text-cyan-600 dark:text-cyan-400 text-lg md:text-2xl font-bold tracking-wide">
          <span className="hidden sm:inline">Physiquest Animation</span>
          <span className="sm:hidden">Physiquest</span>
        </h1>
      </div>

      <div className="flex items-center gap-2 md:gap-4">
        <ToggleSwitch />
      </div>
    </div>
  );
};

export default Headers;

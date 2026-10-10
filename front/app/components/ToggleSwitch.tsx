"use client";

import { useEffect, useState } from "react";

const ToggleSwitch = () => {
  // 初期値は常にfalseでSSRと一致させる
  const [isChecked, setIsChecked] = useState(false);
  const [mounted, setMounted] = useState(false);

  // クライアントマウント後にlocalStorageから読み込み（デフォルトはダーク）
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    // 保存されていない場合はデフォルトでダークモード
    const isDark = savedTheme === null ? true : savedTheme === "dark";
    setIsChecked(isDark);
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    setMounted(true);
  }, []);

  // ダークモードの適用（マウント後のみ）
  useEffect(() => {
    if (!mounted) return;
    if (isChecked) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isChecked, mounted]);

  return (
    <label
      htmlFor="darkModeToggle"
      className={`relative inline-block h-8 w-14 cursor-pointer rounded-full transition-all border ${isChecked ? "bg-slate-700 border-cyan-500/50" : "bg-sky-200 border-cyan-400"}`}
    >
      <input
        type="checkbox"
        id="darkModeToggle"
        className="peer sr-only"
        checked={isChecked}
        onChange={() => setIsChecked(!isChecked)}
      />
      <span
        className={`absolute inset-y-0 start-0 z-10 m-1 inline-flex size-6 items-center justify-center rounded-full transition-all ${isChecked ? 'bg-slate-900 peer-checked:start-6' : 'bg-white'}`}
      >
        {isChecked ? (
          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 stroke-current text-cyan-400" viewBox="0 0 24 24">
            <path fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3h.393a7.5 7.5 0 0 0 7.92 12.446A9 9 0 1 1 12 2.992z"/>
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 fill-current text-yellow-400" viewBox="0 0 24 24">
            <path fillRule="evenodd" d="M12 1.25a.75.75 0 0 1 .75.75v1a.75.75 0 0 1-1.5 0V2a.75.75 0 0 1 .75-.75M4.399 4.399a.75.75 0 0 1 1.06 0l.393.392a.75.75 0 0 1-1.06 1.061l-.393-.393a.75.75 0 0 1 0-1.06m15.202 0a.75.75 0 0 1 0 1.06l-.393.393a.75.75 0 0 1-1.06-1.06l.393-.393a.75.75 0 0 1 1.06 0M12 6.75a5.25 5.25 0 1 0 0 10.5a5.25 5.25 0 0 0 0-10.5M5.25 12a6.75 6.75 0 1 1 13.5 0a6.75 6.75 0 0 1-13.5 0m-4 0a.75.75 0 0 1 .75-.75h1a.75.75 0 0 1 0 1.5H2a.75.75 0 0 1-.75-.75m19 0a.75.75 0 0 1 .75-.75h1a.75.75 0 0 1 0 1.5h-1a.75.75 0 0 1-.75-.75m-2.102 6.148a.75.75 0 0 1 1.06 0l.393.393a.75.75 0 1 1-1.06 1.06l-.393-.393a.75.75 0 0 1 0-1.06m-12.296 0a.75.75 0 0 1 0 1.06l-.393.393a.75.75 0 1 1-1.06-1.06l.392-.393a.75.75 0 0 1 1.061 0M12 20.25a.75.75 0 0 1 .75.75v1a.75.75 0 0 1-1.5 0v-1a.75.75 0 0 1 .75-.75" clipRule="evenodd"/>
          </svg>
        )}
      </span>
    </label>
  );
};

export default ToggleSwitch;

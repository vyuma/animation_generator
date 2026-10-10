"use client";

import { useState } from "react";
import Headers from "@/app/components/header";
import SessionSidebar from "@/app/components/chat/SessionSidebar";
import ChatContainer from "@/app/components/chat/ChatContainer";
import { useVideoGeneration } from "@/app/context/VideoGenerationContext";

export default function Page() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const {
    sessions,
    currentSessionId,
    selectSession,
    createNewSession,
    deleteSession,
    isLoadingSession,
  } = useVideoGeneration();

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);
  const closeSidebar = () => setIsSidebarOpen(false);

  return (
    <div className="flex flex-col h-screen h-[100dvh] bg-slate-100 dark:bg-slate-900 transition-colors">
      <Headers onMenuClick={toggleSidebar} isSidebarOpen={isSidebarOpen} />
      <div className="flex flex-1 overflow-hidden relative">
        {/* サイドバー */}
        <SessionSidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSelectSession={(id) => {
            selectSession(id);
            // モバイルではセッション選択後にサイドバーを閉じる
            if (window.innerWidth < 768) {
              closeSidebar();
            }
          }}
          onNewSession={() => {
            createNewSession();
            // モバイルでは新規作成後にサイドバーを閉じる
            if (window.innerWidth < 768) {
              closeSidebar();
            }
          }}
          onDeleteSession={deleteSession}
          isLoadingSession={isLoadingSession}
          isOpen={isSidebarOpen}
          onClose={closeSidebar}
        />

        {/* メインチャットエリア */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatContainer />
        </div>
      </div>
    </div>
  );
}

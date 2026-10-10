"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  ReactNode,
} from "react";
import {
  JobStatus,
  JobStep,
  ChatMessage,
  GenerationSession,
  GenerationSessionResponse,
  GenerationPromptsResponse,
  PlanInfo,
  SessionPhase,
  SessionIntermediateState,
} from "@/app/type/apiTypes";
import { useMultiJobStream, JobStreamResult } from "@/app/hooks/useMultiJobStream";
import {
  createFullGenerationJob,
  createEditJob,
  getAnimation,
  getGenerationSession,
  getGenerationPrompts,
  createPlan,
  generateFromPlan,
  getJobStatus,
} from "@/app/lib/animationApi";

// generation_idリストのみlocalStorageに保存
const GENERATION_IDS_KEY = "physiquest_generation_ids";
// セッション中間状態を保存するキー
const SESSION_STATE_KEY = "physiquest_session_states";
// 古い状態をクリーンアップする閾値（24時間）
const STATE_CLEANUP_THRESHOLD_MS = 24 * 60 * 60 * 1000;

// セッション中間状態をlocalStorageに保存
const saveSessionState = (state: SessionIntermediateState) => {
  if (typeof window === "undefined") return;
  try {
    const stored = localStorage.getItem(SESSION_STATE_KEY);
    const states: Record<string, SessionIntermediateState> = stored ? JSON.parse(stored) : {};
    states[String(state.generationId)] = state;
    localStorage.setItem(SESSION_STATE_KEY, JSON.stringify(states));
  } catch (e) {
    console.error("Failed to save session state:", e);
  }
};

// 指定generationIdの中間状態を読み込み
const loadSessionState = (generationId: number): SessionIntermediateState | null => {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(SESSION_STATE_KEY);
    if (stored) {
      const states: Record<string, SessionIntermediateState> = JSON.parse(stored);
      return states[String(generationId)] || null;
    }
  } catch (e) {
    console.error("Failed to load session state:", e);
  }
  return null;
};

// 完了したセッションの状態を削除
const clearSessionState = (generationId: number) => {
  if (typeof window === "undefined") return;
  try {
    const stored = localStorage.getItem(SESSION_STATE_KEY);
    if (stored) {
      const states: Record<string, SessionIntermediateState> = JSON.parse(stored);
      delete states[String(generationId)];
      localStorage.setItem(SESSION_STATE_KEY, JSON.stringify(states));
    }
  } catch (e) {
    console.error("Failed to clear session state:", e);
  }
};

// 24時間以上古い状態をクリーンアップ
const cleanupOldStates = () => {
  if (typeof window === "undefined") return;
  try {
    const stored = localStorage.getItem(SESSION_STATE_KEY);
    if (stored) {
      const states: Record<string, SessionIntermediateState> = JSON.parse(stored);
      const now = Date.now();
      let hasChanges = false;
      for (const key of Object.keys(states)) {
        if (now - states[key].updatedAt > STATE_CLEANUP_THRESHOLD_MS) {
          delete states[key];
          hasChanges = true;
        }
      }
      if (hasChanges) {
        localStorage.setItem(SESSION_STATE_KEY, JSON.stringify(states));
      }
    }
  } catch (e) {
    console.error("Failed to cleanup old states:", e);
  }
};

interface VideoGenerationState {
  sessions: GenerationSession[];
  currentSessionId: number | null;
  isLoading: boolean;
  isLoadingSession: boolean;
}

interface VideoGenerationActions {
  createNewSession: () => void;
  selectSession: (sessionId: number) => Promise<void>;
  deleteSession: (sessionId: number) => void;
  sendMessage: (content: string, usePlanning?: boolean) => Promise<void>;
  editVideo: (enhancePrompt: string) => Promise<void>;
  getCurrentSession: () => GenerationSession | null;
  loadSessionFromServer: (generationId: number) => Promise<void>;
  addGenerationId: (generationId: number) => void;
  // プランニング関連
  confirmPlan: () => Promise<void>;
  editPlan: () => void;
  updatePlan: (content: string) => void;
  cancelPlan: () => void;
}

type VideoGenerationContextType = VideoGenerationState & VideoGenerationActions;

const VideoGenerationContext = createContext<
  VideoGenerationContextType | undefined
>(undefined);

// generation_idリストをlocalStorageから読み込み
const loadGenerationIds = (): number[] => {
  if (typeof window === "undefined") return [];
  try {
    const stored = localStorage.getItem(GENERATION_IDS_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (e) {
    console.error("Failed to load generation IDs:", e);
  }
  return [];
};

// generation_idリストをlocalStorageに保存
const saveGenerationIds = (ids: number[]) => {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(GENERATION_IDS_KEY, JSON.stringify(ids));
  } catch (e) {
    console.error("Failed to save generation IDs:", e);
  }
};

// ユニークID生成
const generateId = () => Math.random().toString(36).substr(2, 9);

// 一時セッションID生成（負の数を使用してサーバーのIDと区別）
let tempSessionCounter = -1;
const generateTempSessionId = () => {
  tempSessionCounter -= 1;
  return tempSessionCounter;
};

// APIレスポンスからChatMessageを構築
const buildMessagesFromApiResponse = async (
  sessionResponse: GenerationSessionResponse,
  promptsResponse: GenerationPromptsResponse
): Promise<{ messages: ChatMessage[]; videoUrl?: string; videoId?: string }> => {
  const messages: ChatMessage[] = [];
  let latestVideoUrl: string | undefined;
  let latestVideoId: string | undefined;

  // 動画をedit_countでソート（昇順: 0, 1, 2...）
  const sortedVideos = [...sessionResponse.videos].sort((a, b) => a.edit_count - b.edit_count);

  // 各動画に対応するプロンプトを探して、プロンプト→動画の順で追加
  for (const video of sortedVideos) {
    // edit_countに対応するプロンプトを探す (edit_count 0 → trial 1, edit_count 1 → trial 2, ...)
    const prompt = promptsResponse.prompts.find(p => p.trial === video.edit_count + 1);

    if (prompt) {
      // contentとenhance_promptを組み合わせてメッセージを構築
      let userContent = "";

      if (prompt.trial === 1) {
        // 初回生成: まずcontentを取得、次にenhance_promptを取得
        if (prompt.content) {
          userContent = prompt.content;
        }
        // enhance_promptがあり、contentと異なる場合は追加
        if (prompt.enhance_prompt && prompt.enhance_prompt !== prompt.content) {
          if (userContent) {
            userContent = `${userContent}\n\n【生成プロンプト】\n${prompt.enhance_prompt}`;
          } else {
            userContent = prompt.enhance_prompt;
          }
        }
      } else {
        // 編集: まずcontentを取得、次にenhance_promptを取得
        if (prompt.content) {
          userContent = `User: ${prompt.content}`;
        }
        // enhance_promptがあり、contentと異なる場合は追加
        if (prompt.enhance_prompt && prompt.enhance_prompt !== prompt.content) {
          if (userContent) {
            userContent = `${userContent}\n\n【生成プロンプト】\n${prompt.enhance_prompt}`;
          } else {
            userContent = `User: ${prompt.enhance_prompt}`;
          }
        }
      }

      if (userContent) {
        messages.push({
          id: generateId(),
          type: "user",
          content: userContent,
          timestamp: video.generate_time ? new Date(video.generate_time) : new Date(),
        });
      }
    }

    // 動画メッセージを追加
    try {
      const videoUrl = await getAnimation(video.video_id);
      messages.push({
        id: generateId(),
        type: "video",
        content: video.edit_count === 0 ? "動画が生成されました" : "編集が完了しました",
        timestamp: video.generate_time ? new Date(video.generate_time) : new Date(),
        videoUrl,
        videoId: video.video_id,
      });

      if (video.video_id === sessionResponse.latest_video_id) {
        latestVideoUrl = videoUrl;
        latestVideoId = video.video_id;
      }
    } catch (error) {
      console.error(`Failed to load video ${video.video_id}:`, error);
    }
  }

  return { messages, videoUrl: latestVideoUrl, videoId: latestVideoId };
};

export const VideoGenerationProvider = ({
  children,
}: {
  children: ReactNode;
}) => {
  const [sessions, setSessions] = useState<GenerationSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [generationIds, setGenerationIds] = useState<number[]>([]);

  // ロード中/ロード済みのセッションIDを追跡（無限ループ防止）
  const loadingSessionsRef = useRef<Set<number>>(new Set());
  const loadedSessionsRef = useRef<Set<number>>(new Set());

  // セッションをjobIdで検索するヘルパー
  const findSessionByJobId = useCallback((jobId: string) => {
    return sessions.find((s) => s.jobId === jobId);
  }, [sessions]);

  // currentSessionIdをrefで追跡（コールバック内で参照するため）
  const currentSessionIdRef = useRef(currentSessionId);
  currentSessionIdRef.current = currentSessionId;

  // 進捗更新のハンドラー
  const handleProgress = useCallback((result: JobStreamResult) => {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.jobId === result.jobId) {
          const progressMsgIndex = s.messages.findIndex((m) => m.type === "progress");
          const newMessages = [...s.messages];
          const progressMsg: ChatMessage = {
            id: progressMsgIndex >= 0 ? s.messages[progressMsgIndex].id : generateId(),
            type: "progress",
            content: result.message || "生成中...",
            timestamp: new Date(),
            status: result.status || undefined,
            currentStep: result.currentStep || undefined,
            progress: result.progress,
          };

          if (progressMsgIndex >= 0) {
            newMessages[progressMsgIndex] = progressMsg;
          } else {
            newMessages.push(progressMsg);
          }

          return {
            ...s,
            status: result.status || s.status,
            messages: newMessages,
          };
        }
        return s;
      })
    );
  }, []);

  // 完了ハンドラー
  const handleCompleted = useCallback(async (result: JobStreamResult) => {
    console.log("Job completed:", result);

    if (!result.videoId) {
      console.error("No video ID in completed result");
      return;
    }

    const jobId = result.jobId;
    const videoId = result.videoId;
    const newGenerationId = result.generationId;

    // まずセッションを「動画取得中」状態に更新
    setSessions((prev) =>
      prev.map((s) => {
        if (s.jobId === jobId) {
          const progressMsgIndex = s.messages.findIndex((m) => m.type === "progress");
          const newMessages = [...s.messages];
          if (progressMsgIndex >= 0) {
            newMessages[progressMsgIndex] = {
              ...newMessages[progressMsgIndex],
              content: "動画を取得中...",
              progress: 100,
            };
          }
          return { ...s, messages: newMessages };
        }
        return s;
      })
    );

    try {
      console.log("Fetching video:", videoId);
      const videoUrl = await getAnimation(videoId);
      console.log("Video URL obtained:", videoUrl);

      // generation_idリストに追加
      if (newGenerationId && newGenerationId > 0) {
        setGenerationIds((prev) => {
          if (!prev.includes(newGenerationId)) {
            return [newGenerationId, ...prev];
          }
          return prev;
        });

        // 中間状態をクリア
        clearSessionState(newGenerationId);
      }

      // 現在のセッションIDが変更される場合に備えて追跡
      let oldSessionId: number | null = null;
      let needsCurrentSessionUpdate = false;

      setSessions((prev) => {
        // 該当セッションを見つけて、currentSessionIdの更新が必要か確認
        const targetSession = prev.find((s) => s.jobId === jobId);
        if (targetSession && targetSession.generationId === currentSessionIdRef.current) {
          if (newGenerationId && newGenerationId !== targetSession.generationId) {
            oldSessionId = targetSession.generationId;
            needsCurrentSessionUpdate = true;
          }
        }

        return prev.map((s) => {
          if (s.jobId === jobId) {
            const messagesWithoutProgress = s.messages.filter((m) => m.type !== "progress");
            // 既に動画があるセッションへの完了通知は「編集」、初回は「生成」
            const isEdit = Boolean(s.videoId);

            const newMessage: ChatMessage = {
              id: generateId(),
              type: "video",
              content: isEdit ? "編集が完了しました" : "動画が生成されました",
              timestamp: new Date(),
              videoUrl,
              videoId,
            };

            // 中間状態をクリア（既存のgenerationIdの場合）
            if (s.generationId && s.generationId > 0) {
              clearSessionState(s.generationId);
            }

            const updatedSession: GenerationSession = {
              ...s,
              generationId: newGenerationId ?? s.generationId,
              status: "completed" as const,
              phase: "completed" as const,
              videoUrl,
              videoId,
              jobId: null,
              messages: [...messagesWithoutProgress, newMessage],
            };

            return updatedSession;
          }
          return s;
        });
      });

      // currentSessionIdを新しいIDに更新
      if (needsCurrentSessionUpdate && newGenerationId) {
        setCurrentSessionId(newGenerationId);
      }
    } catch (error) {
      console.error("Failed to fetch video:", error);
      // エラー時もセッションを更新
      let errorNeedsCurrentSessionUpdate = false;

      setSessions((prev) => {
        // 該当セッションを見つけて、currentSessionIdの更新が必要か確認
        const targetSession = prev.find((s) => s.jobId === jobId);
        if (targetSession && targetSession.generationId === currentSessionIdRef.current) {
          if (newGenerationId && newGenerationId !== targetSession.generationId) {
            errorNeedsCurrentSessionUpdate = true;
          }
        }

        return prev.map((s) => {
          if (s.jobId === jobId) {
            const messagesWithoutProgress = s.messages.filter((m) => m.type !== "progress");

            // 中間状態をクリア
            if (s.generationId && s.generationId > 0) {
              clearSessionState(s.generationId);
            }

            const errorMessage: ChatMessage = {
              id: generateId(),
              type: "error" as const,
              content: "動画の取得に失敗しました",
              timestamp: new Date(),
            };

            return {
              ...s,
              generationId: newGenerationId ?? s.generationId,
              status: "failed" as const,
              phase: "failed" as const,
              jobId: null,
              messages: [...messagesWithoutProgress, errorMessage],
            };
          }
          return s;
        });
      });

      // currentSessionIdを新しいIDに更新（エラー時）
      if (errorNeedsCurrentSessionUpdate && newGenerationId) {
        setCurrentSessionId(newGenerationId);
      }
    }
  }, []); // refを使用しているため安定

  // エラーハンドラー
  const handleError = useCallback((result: JobStreamResult) => {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.jobId === result.jobId) {
          const messagesWithoutProgress = s.messages.filter((m) => m.type !== "progress");

          // 中間状態をクリア
          if (s.generationId && s.generationId > 0) {
            clearSessionState(s.generationId);
          }

          const errorMessage: ChatMessage = {
            id: generateId(),
            type: "error" as const,
            content: result.error || "エラーが発生しました",
            timestamp: new Date(),
          };

          return {
            ...s,
            status: "failed" as const,
            phase: "failed" as const,
            jobId: null,
            messages: [...messagesWithoutProgress, errorMessage],
          };
        }
        return s;
      })
    );
  }, []); // refを使用しているため安定

  // マルチジョブストリーム
  const { startWatching, stopWatching } = useMultiJobStream({
    onProgress: handleProgress,
    onCompleted: handleCompleted,
    onError: handleError,
  });

  // startWatchingをrefで追跡
  const startWatchingRef = useRef(startWatching);
  startWatchingRef.current = startWatching;

  // 初回マウント時の処理（一度だけ実行）: localStorageから読み込み
  useEffect(() => {
    if (mounted) return; // 既にマウント済みならスキップ

    const ids = loadGenerationIds();
    setGenerationIds(ids);

    const initialSessions: GenerationSession[] = ids.map((id) => ({
      generationId: id,
      title: `セッション ${id}`,
      createdAt: new Date(),
      messages: [],
      status: "idle" as const,
      jobId: null,
    }));
    setSessions(initialSessions);

    if (ids.length > 0) {
      setCurrentSessionId(ids[0]);
    }
    setMounted(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // mountedは意図的に除外

  // generation_idリストの変更をlocalStorageに保存（未ログイン時のみ）
  useEffect(() => {
    if (mounted) {
      // 正のIDのみ保存（一時セッションは除外）
      const validIds = generationIds.filter((id) => id > 0);
      saveGenerationIds(validIds);
    }
  }, [generationIds, mounted]);

  // セッション中間状態の自動保存（未ログイン時のみ）
  useEffect(() => {
    if (!mounted) return;

    // 各セッションの中間状態を保存
    sessions.forEach((session) => {
      if (!session.generationId || session.generationId < 0) return;

      const phase = session.phase || "idle";

      // 中間状態（idle, completed以外）を保存
      if (phase !== "idle" && phase !== "completed") {
        saveSessionState({
          generationId: session.generationId,
          phase,
          plan: session.plan || null,
          jobId: session.jobId || null,
          updatedAt: Date.now(),
        });
      } else if (phase === "completed" || phase === "idle") {
        // 完了またはアイドル状態になったら状態を削除
        clearSessionState(session.generationId);
      }
    });
  }, [sessions, mounted]);

  // マウント時に古い状態をクリーンアップ
  useEffect(() => {
    if (mounted) {
      cleanupOldStates();
    }
  }, [mounted]);

  // サーバーからセッション情報を読み込み（中間状態の復元を含む）
  const loadSessionFromServer = useCallback(async (generationId: number) => {
    if (generationId < 0) return; // 一時セッションは読み込まない

    // 既にロード中またはロード済みならスキップ
    if (loadingSessionsRef.current.has(generationId) || loadedSessionsRef.current.has(generationId)) {
      return;
    }

    // ロード中としてマーク
    loadingSessionsRef.current.add(generationId);
    setIsLoadingSession(true);
    try {
      // 1. localStorageから中間状態を復元
      const savedState = loadSessionState(generationId);

      // 2. サーバーからセッション情報を取得
      const [sessionResponse, promptsResponse] = await Promise.all([
        getGenerationSession(generationId),
        getGenerationPrompts(generationId),
      ]);

      const { messages, videoUrl, videoId } = await buildMessagesFromApiResponse(
        sessionResponse,
        promptsResponse
      );

      const title = promptsResponse.prompts[0]?.content?.slice(0, 30) ||
        promptsResponse.prompts[0]?.enhance_prompt?.slice(0, 30) ||
        `セッション ${generationId}`;

      // 3. 中間状態とジョブステータスを確認
      let finalPhase: SessionPhase = "completed";
      let finalPlan: PlanInfo | null = null;
      let finalJobId: string | null = null;
      let finalStatus: JobStatus | "idle" = "completed";
      let additionalMessages: ChatMessage[] = [];

      if (savedState) {
        const { phase, plan, jobId } = savedState;

        // ジョブが保存されていれば現在のステータスを確認
        if (jobId) {
          try {
            const jobStatus = await getJobStatus(jobId);

            if (jobStatus.status === "pending" || jobStatus.status === "running") {
              // ジョブが実行中ならSSE監視を再開
              finalPhase = phase;
              finalPlan = plan;
              finalJobId = jobId;
              finalStatus = jobStatus.status;

              // 進捗メッセージを追加
              additionalMessages.push({
                id: generateId(),
                type: "progress" as const,
                content: jobStatus.message || "生成中...",
                timestamp: new Date(),
                status: jobStatus.status,
                currentStep: jobStatus.current_step,
                progress: jobStatus.progress,
              });

              // SSE監視を再開
              startWatchingRef.current(jobId);
            } else if (jobStatus.status === "completed" && jobStatus.result?.video_id) {
              // ジョブが完了していれば動画を取得（messages/videoUrlは既に取得済みのはず）
              finalPhase = "completed";
              clearSessionState(generationId);
            } else if (jobStatus.status === "failed") {
              // ジョブが失敗していれば失敗状態に
              finalPhase = "failed";
              finalStatus = "failed";
              clearSessionState(generationId);
              additionalMessages.push({
                id: generateId(),
                type: "error" as const,
                content: jobStatus.error || "生成に失敗しました",
                timestamp: new Date(),
              });
            }
          } catch {
            // ジョブが見つからない場合は状態をクリア
            console.warn(`Job ${jobId} not found, clearing saved state`);
            clearSessionState(generationId);
          }
        } else if (phase === "plan_review" && plan) {
          // プランレビュー中（ジョブなし）
          finalPhase = "plan_review";
          finalPlan = plan;
          finalStatus = "idle";

          // プランレビュー中のシステムメッセージを追加
          additionalMessages.push({
            id: generateId(),
            type: "system" as const,
            content: "プランが作成されました。確認または編集してください。",
            timestamp: new Date(),
          });
        } else if (phase === "planning") {
          // プランニング中だった場合は状態をリセット（プランニングAPIは同期なので再実行不可）
          finalPhase = "idle";
          clearSessionState(generationId);
        }
      }

      // 動画がある場合は完了状態
      if (videoId && videoUrl) {
        finalPhase = "completed";
        finalStatus = "completed";
      }

      const session: GenerationSession = {
        generationId,
        title: title + (title.length >= 30 ? "..." : ""),
        createdAt: sessionResponse.generate_time
          ? new Date(sessionResponse.generate_time)
          : new Date(),
        messages: [...messages, ...additionalMessages],
        videoUrl,
        videoId,
        status: finalStatus,
        jobId: finalJobId,
        phase: finalPhase,
        plan: finalPlan,
      };

      setSessions((prev) => {
        const existing = prev.findIndex((s) => s.generationId === generationId);
        if (existing >= 0) {
          const updated = [...prev];
          updated[existing] = session;
          return updated;
        }
        return [session, ...prev];
      });
    } catch (error) {
      console.error("Failed to load session from server:", error);
      setSessions((prev) =>
        prev.map((s) => {
          if (s.generationId === generationId) {
            return {
              ...s,
              status: "failed" as const,
              messages: [
                {
                  id: generateId(),
                  type: "error" as const,
                  content: "セッションの読み込みに失敗しました",
                  timestamp: new Date(),
                },
              ],
            };
          }
          return s;
        })
      );
    } finally {
      // ロード完了としてマーク
      loadingSessionsRef.current.delete(generationId);
      loadedSessionsRef.current.add(generationId);
      setIsLoadingSession(false);
    }
  }, []); // refを使用しているため安定

  // currentSessionIdが変更されたときにサーバーからセッションを読み込み
  useEffect(() => {
    if (!mounted || !currentSessionId || currentSessionId < 0) return;

    // 既にロード中またはロード済みならスキップ（refで管理）
    if (loadingSessionsRef.current.has(currentSessionId) || loadedSessionsRef.current.has(currentSessionId)) {
      return;
    }

    // sessionsの状態をチェック（setSessionsのコールバック内でアクセス）
    setSessions((prev) => {
      const session = prev.find((s) => s.generationId === currentSessionId);
      if (session && session.messages.length === 0 && !session.jobId) {
        // 非同期でロードを開始（状態更新の外で実行）
        setTimeout(() => loadSessionFromServer(currentSessionId), 0);
      }
      return prev; // 状態は変更しない
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, mounted]);

  // generation_idを追加
  const addGenerationId = useCallback((generationId: number) => {
    if (generationId > 0 && !generationIds.includes(generationId)) {
      setGenerationIds((prev) => [generationId, ...prev]);
    }
  }, [generationIds]);

  // 新規セッション作成
  const createNewSession = useCallback(() => {
    const tempId = generateTempSessionId();
    const newSession: GenerationSession = {
      generationId: tempId,
      title: "",
      createdAt: new Date(),
      messages: [],
      status: "idle",
      jobId: null,
    };

    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(tempId);
  }, []);

  // セッション選択
  const selectSession = useCallback(async (sessionId: number) => {
    setCurrentSessionId(sessionId);

    if (sessionId > 0) {
      const session = sessions.find((s) => s.generationId === sessionId);
      if (session && session.messages.length === 0 && !session.jobId) {
        await loadSessionFromServer(sessionId);
      }
    }
  }, [sessions, loadSessionFromServer]);

  // セッション削除
  const deleteSession = useCallback(
    async (sessionId: number) => {
      // 実行中のジョブがあれば監視を停止
      const session = sessions.find((s) => s.generationId === sessionId);
      if (session?.jobId) {
        stopWatching(session.jobId);
      }

      setSessions((prev) => prev.filter((s) => s.generationId !== sessionId));
      if (sessionId > 0) {
        setGenerationIds((prev) => prev.filter((id) => id !== sessionId));

      }

      if (currentSessionId === sessionId) {
        const remaining = sessions.filter((s) => s.generationId !== sessionId);
        setCurrentSessionId(remaining.length > 0 ? remaining[0].generationId : null);
      }
    },
    [currentSessionId, sessions, stopWatching]
  );

  // 現在のセッションを取得
  const getCurrentSession = useCallback((): GenerationSession | null => {
    return sessions.find((s) => s.generationId === currentSessionId) || null;
  }, [sessions, currentSessionId]);

  // メッセージ送信（動画生成または編集）
  const sendMessage = useCallback(
    async (content: string, usePlanning: boolean = false) => {
      let targetSessionId = currentSessionId;
      let targetSession = sessions.find((s) => s.generationId === targetSessionId);

      // 現在のセッションに動画がある場合は編集モード
      if (targetSession && targetSession.videoId && targetSession.generationId && targetSession.generationId > 0) {
        // editVideoを呼び出す代わりに、直接編集処理を実行
        const sessionId = targetSession.generationId;
        const videoId = targetSession.videoId;

        // ユーザーメッセージを追加
        const userMessage: ChatMessage = {
          id: generateId(),
          type: "user",
          content: `User: ${content}`,
          timestamp: new Date(),
        };

        setSessions((prev) =>
          prev.map((s) => {
            if (s.generationId === sessionId) {
              return {
                ...s,
                messages: [...s.messages, userMessage],
              };
            }
            return s;
          })
        );

        try {
          const response = await createEditJob({
            generation_id: sessionId,
            prior_video_id: videoId,
            enhance_prompt: content,
          });

          const jobId = response.job_id;

          setSessions((prev) =>
            prev.map((s) => {
              if (s.generationId === sessionId) {
                return {
                  ...s,
                  status: "pending" as const,
                  phase: "generating" as const,
                  jobId,
                  messages: [
                    ...s.messages,
                    {
                      id: generateId(),
                      type: "system" as const,
                      content: "動画を編集中...",
                      timestamp: new Date(),
                    },
                  ],
                };
              }
              return s;
            })
          );

          startWatching(jobId);
        } catch (error) {
          const errorMessage: ChatMessage = {
            id: generateId(),
            type: "error" as const,
            content: error instanceof Error ? error.message : "編集に失敗しました",
            timestamp: new Date(),
          };

          setSessions((prev) =>
            prev.map((s) => {
              if (s.generationId === sessionId) {
                return {
                  ...s,
                  status: "failed" as const,
                  messages: [...s.messages, errorMessage],
                };
              }
              return s;
            })
          );
        }
        return;
      }

      // 現在のセッションがない場合は新規作成
      if (!targetSession) {
        const tempId = generateTempSessionId();
        const newSession: GenerationSession = {
          generationId: tempId,
          title: content.slice(0, 30) + (content.length > 30 ? "..." : ""),
          createdAt: new Date(),
          messages: [],
          status: "idle",
          jobId: null,
        };
        setSessions((prev) => [newSession, ...prev]);
        targetSessionId = tempId;
        setCurrentSessionId(tempId);
      }

      // ユーザーメッセージを追加
      const userMessage: ChatMessage = {
        id: generateId(),
        type: "user",
        content,
        timestamp: new Date(),
      };

      setSessions((prev) =>
        prev.map((s) => {
          if (s.generationId === targetSessionId) {
            const newTitle = s.title || content.slice(0, 30) + (content.length > 30 ? "..." : "");
            return {
              ...s,
              title: newTitle,
              messages: [...s.messages, userMessage],
            };
          }
          return s;
        })
      );

      if (usePlanning) {
        // プランニングモード
        setSessions((prev) =>
          prev.map((s) => {
            if (s.generationId === targetSessionId) {
              return {
                ...s,
                phase: "planning" as const,
                messages: [
                  ...s.messages,
                  {
                    id: generateId(),
                    type: "system" as const,
                    content: "プランを作成中...",
                    timestamp: new Date(),
                  },
                ],
              };
            }
            return s;
          })
        );

        try {
          const planResponse = await createPlan(content);

          // generation_idをリストに追加
          if (planResponse.generation_id && planResponse.generation_id > 0) {
            setGenerationIds((prev) => {
              if (!prev.includes(planResponse.generation_id!)) {
                return [planResponse.generation_id!, ...prev];
              }
              return prev;
            });

          }

          // 一時IDから正式IDに更新
          const newGenerationId = planResponse.generation_id ?? targetSessionId;

          // currentSessionIdも更新（重要：これがないとセッションが見つからなくなる）
          if (newGenerationId !== targetSessionId) {
            setCurrentSessionId(newGenerationId);
          }

          setSessions((prev) =>
            prev.map((s) => {
              if (s.generationId === targetSessionId) {
                const messagesWithoutSystem = s.messages.filter((m) =>
                  !(m.type === "system" && m.content === "プランを作成中...")
                );
                return {
                  ...s,
                  generationId: newGenerationId,
                  phase: "plan_review" as const,
                  plan: {
                    content: planResponse.plan,
                    generationId: newGenerationId as number,
                    isEditing: false,
                  },
                  messages: [
                    ...messagesWithoutSystem,
                    {
                      id: generateId(),
                      type: "system" as const,
                      content: "プランが作成されました。確認または編集してください。",
                      timestamp: new Date(),
                    },
                  ],
                };
              }
              return s;
            })
          );
        } catch (error) {
          setSessions((prev) =>
            prev.map((s) => {
              if (s.generationId === targetSessionId) {
                return {
                  ...s,
                  status: "failed" as const,
                  phase: "failed" as const,
                  messages: [
                    ...s.messages.filter((m) =>
                      !(m.type === "system" && m.content === "プランを作成中...")
                    ),
                    {
                      id: generateId(),
                      type: "error" as const,
                      content: error instanceof Error ? error.message : "プランニングに失敗しました",
                      timestamp: new Date(),
                    },
                  ],
                };
              }
              return s;
            })
          );
        }
      } else {
        // フル生成モード
        try {
          const response = await createFullGenerationJob(content);
          const jobId = response.job_id;

          // job_idをキーとしてセッションを保存（generation_idが確定したら移行）

          // セッションにジョブIDを設定してSSE監視開始
          setSessions((prev) =>
            prev.map((s) => {
              if (s.generationId === targetSessionId) {
                return {
                  ...s,
                  status: "pending" as const,
                  phase: "generating" as const,
                  jobId,
                  messages: [
                    ...s.messages,
                    {
                      id: generateId(),
                      type: "system" as const,
                      content: "動画生成を開始しました...",
                      timestamp: new Date(),
                    },
                  ],
                };
              }
              return s;
            })
          );

          startWatching(jobId);
        } catch (error) {
          setSessions((prev) =>
            prev.map((s) => {
              if (s.generationId === targetSessionId) {
                return {
                  ...s,
                  status: "failed" as const,
                  phase: "failed" as const,
                  messages: [
                    ...s.messages,
                    {
                      id: generateId(),
                      type: "error" as const,
                      content: error instanceof Error ? error.message : "生成に失敗しました",
                      timestamp: new Date(),
                    },
                  ],
                };
              }
              return s;
            })
          );
        }
      }
    },
    [currentSessionId, sessions, startWatching]
  );

  // 動画編集
  const editVideo = useCallback(
    async (enhancePrompt: string) => {
      const currentSession = getCurrentSession();
      if (!currentSession || !currentSession.videoId || !currentSession.generationId || currentSession.generationId < 0) {
        console.error("No valid session for editing");
        return;
      }

      const sessionId = currentSession.generationId;

      // ユーザーメッセージを追加
      const userMessage: ChatMessage = {
        id: generateId(),
        type: "user",
        content: `User: ${enhancePrompt}`,
        timestamp: new Date(),
      };

      setSessions((prev) =>
        prev.map((s) => {
          if (s.generationId === sessionId) {
            return {
              ...s,
              messages: [...s.messages, userMessage],
            };
          }
          return s;
        })
      );

      try {
        const response = await createEditJob({
          generation_id: currentSession.generationId,
          prior_video_id: currentSession.videoId,
          enhance_prompt: enhancePrompt,
        });

        const jobId = response.job_id;

        setSessions((prev) =>
          prev.map((s) => {
            if (s.generationId === sessionId) {
              return {
                ...s,
                status: "pending" as const,
                phase: "generating" as const,
                jobId,
                messages: [
                  ...s.messages,
                  {
                    id: generateId(),
                    type: "system" as const,
                    content: "動画を編集中...",
                    timestamp: new Date(),
                  },
                ],
              };
            }
            return s;
          })
        );

        startWatching(jobId);
      } catch (error) {
        setSessions((prev) =>
          prev.map((s) => {
            if (s.generationId === sessionId) {
              return {
                ...s,
                status: "failed" as const,
                messages: [
                  ...s.messages,
                  {
                    id: generateId(),
                    type: "error" as const,
                    content: error instanceof Error ? error.message : "編集に失敗しました",
                    timestamp: new Date(),
                  },
                ],
              };
            }
            return s;
          })
        );
      }
    },
    [getCurrentSession, startWatching]
  );

  // プランを確認して動画生成を開始
  const confirmPlan = useCallback(async () => {
    const currentSession = getCurrentSession();
    if (!currentSession || !currentSession.plan || !currentSession.generationId) {
      console.error("No valid session or plan for confirmation");
      return;
    }

    const sessionId = currentSession.generationId;
    const planContent = currentSession.plan.content;

    // プランメッセージを作成
    const planMessage: ChatMessage = {
      id: generateId(),
      type: "plan" as const,
      content: "確定されたプラン",
      planContent: planContent,
      timestamp: new Date(),
    };

    // 生成中状態に更新（プランをメッセージとして保存）
    setSessions((prev) =>
      prev.map((s) => {
        if (s.generationId === sessionId) {
          return {
            ...s,
            phase: "generating" as const,
            plan: null, // プランはメッセージに保存されるのでnullに
            messages: [
              ...s.messages,
              planMessage,
              {
                id: generateId(),
                type: "system" as const,
                content: "プランを確認しました。動画を生成中...",
                timestamp: new Date(),
              },
            ],
          };
        }
        return s;
      })
    );

    try {
      const response = await generateFromPlan(sessionId, planContent);

      if (response.video_id) {
        // 同期APIの場合、直接動画取得
        const videoUrl = await getAnimation(response.video_id);

        const videoMessage: ChatMessage = {
          id: generateId(),
          type: "video" as const,
          content: "動画が生成されました",
          timestamp: new Date(),
          videoUrl,
          videoId: response.video_id,
        };

        setSessions((prev) =>
          prev.map((s) => {
            if (s.generationId === sessionId) {
              const messagesFiltered = s.messages.filter((m) =>
                !(m.type === "system" && m.content === "プランを確認しました。動画を生成中...")
              );
              return {
                ...s,
                status: "completed" as const,
                phase: "completed" as const,
                videoUrl,
                videoId: response.video_id,
                messages: [...messagesFiltered, videoMessage],
              };
            }
            return s;
          })
        );

      } else if (response.job_id) {
        // 非同期ジョブの場合

        setSessions((prev) =>
          prev.map((s) => {
            if (s.generationId === sessionId) {
              return {
                ...s,
                status: "pending" as const,
                jobId: response.job_id,
              };
            }
            return s;
          })
        );
        startWatching(response.job_id);
      }
    } catch (error) {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.generationId === sessionId) {
            return {
              ...s,
              status: "failed" as const,
              phase: "failed" as const,
              messages: [
                ...s.messages.filter((m) =>
                  !(m.type === "system" && m.content === "プランを確認しました。動画を生成中...")
                ),
                {
                  id: generateId(),
                  type: "error" as const,
                  content: error instanceof Error ? error.message : "動画生成に失敗しました",
                  timestamp: new Date(),
                },
              ],
            };
          }
          return s;
        })
      );
    }
  }, [getCurrentSession, startWatching]);

  // プラン編集モードに入る
  const editPlan = useCallback(() => {
    const currentSession = getCurrentSession();
    if (!currentSession || !currentSession.plan) return;

    const sessionId = currentSession.generationId;

    setSessions((prev) =>
      prev.map((s) => {
        if (s.generationId === sessionId && s.plan) {
          return {
            ...s,
            plan: { ...s.plan, isEditing: true },
          };
        }
        return s;
      })
    );
  }, [getCurrentSession]);

  // プラン内容を更新
  const updatePlan = useCallback((content: string) => {
    const currentSession = getCurrentSession();
    if (!currentSession || !currentSession.plan) return;

    const sessionId = currentSession.generationId;

    setSessions((prev) =>
      prev.map((s) => {
        if (s.generationId === sessionId && s.plan) {
          const updatedPlan = { ...s.plan, content, isEditing: false };

          return {
            ...s,
            plan: updatedPlan,
          };
        }
        return s;
      })
    );
  }, [getCurrentSession]);

  // プランをキャンセル
  const cancelPlan = useCallback(() => {
    const currentSession = getCurrentSession();
    if (!currentSession) return;

    const sessionId = currentSession.generationId;

    const cancelMessage: ChatMessage = {
      id: generateId(),
      type: "system" as const,
      content: "プランをキャンセルしました。",
      timestamp: new Date(),
    };

    setSessions((prev) =>
      prev.map((s) => {
        if (s.generationId === sessionId) {
          return {
            ...s,
            phase: "idle" as const,
            plan: null,
            messages: [...s.messages, cancelMessage],
          };
        }
        return s;
      })
    );

  }, [getCurrentSession]);

  // isLoadingは現在のセッションがジョブ実行中かどうか
  const currentSession = sessions.find((s) => s.generationId === currentSessionId);
  const isCurrentLoading = currentSession?.jobId != null;

  const value: VideoGenerationContextType = {
    sessions,
    currentSessionId,
    isLoading: isCurrentLoading,
    isLoadingSession,
    createNewSession,
    selectSession,
    deleteSession,
    sendMessage,
    editVideo,
    getCurrentSession,
    loadSessionFromServer,
    addGenerationId,
    // プランニング関連
    confirmPlan,
    editPlan,
    updatePlan,
    cancelPlan,
  };

  return (
    <VideoGenerationContext.Provider value={value}>
      {children}
    </VideoGenerationContext.Provider>
  );
};

export const useVideoGeneration = () => {
  const context = useContext(VideoGenerationContext);
  if (!context) {
    throw new Error(
      "useVideoGeneration must be used within a VideoGenerationProvider"
    );
  }
  return context;
};

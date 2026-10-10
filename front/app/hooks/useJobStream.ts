"use client";

import { useEffect, useState } from "react";
import { JobStatus, JobStep } from "@/app/type/apiTypes";
import { getJobStreamUrl } from "@/app/lib/animationApi";

interface UseJobStreamResult {
  status: JobStatus | null;
  currentStep: JobStep | null;
  progress: number;
  videoId: string | null;
  generationId: number | null;
  error: string | null;
  isConnected: boolean;
  message: string | null;
}

export function useJobStream(jobId: string | null): UseJobStreamResult {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [currentStep, setCurrentStep] = useState<JobStep | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [generationId, setGenerationId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    // SSR時はEventSourceを使用しない
    if (typeof window === "undefined") return;

    if (!jobId) {
      // jobIdがない場合は状態をリセット
      setStatus(null);
      setCurrentStep(null);
      setProgress(0);
      setVideoId(null);
      setGenerationId(null);
      setError(null);
      setIsConnected(false);
      setMessage(null);
      return;
    }

    const streamUrl = getJobStreamUrl(jobId);
    const eventSource = new EventSource(streamUrl);

    eventSource.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    // 進捗イベント
    eventSource.addEventListener("progress", (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus(data.status);
        setCurrentStep(data.current_step);
        setProgress(data.progress * 100); // 0-1 -> 0-100
        if (data.message) {
          setMessage(data.message);
        }
      } catch (e) {
        console.error("Progress event parse error:", e);
      }
    });

    // 完了イベント
    eventSource.addEventListener("completed", (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus("completed");
        setVideoId(data.video_id);
        setGenerationId(data.generation_id);
        setProgress(100);
        if (data.message) {
          setMessage(data.message);
        }
        eventSource.close();
        setIsConnected(false);
      } catch (e) {
        console.error("Completed event parse error:", e);
      }
    });

    // エラーイベント
    eventSource.addEventListener("error", (event) => {
      // SSEのエラーイベント（接続エラーなど）
      if (eventSource.readyState === EventSource.CLOSED) {
        setIsConnected(false);
      }
    });

    // カスタムエラーイベント（サーバーからのエラー通知）
    eventSource.addEventListener("job_error", (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus("failed");
        setError(data.error || "ジョブの実行中にエラーが発生しました");
        eventSource.close();
        setIsConnected(false);
      } catch (e) {
        console.error("Error event parse error:", e);
        setError("不明なエラーが発生しました");
      }
    });

    // デフォルトメッセージイベント
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status) {
          setStatus(data.status);
        }
        if (data.current_step) {
          setCurrentStep(data.current_step);
        }
        if (data.progress !== undefined) {
          // 進捗は0-1か0-100のどちらでも対応
          const progressValue = data.progress <= 1 ? data.progress * 100 : data.progress;
          setProgress(progressValue);
        }
        if (data.video_id) {
          setVideoId(data.video_id);
        }
        if (data.generation_id) {
          setGenerationId(data.generation_id);
        }
        if (data.error) {
          setError(data.error);
        }
        if (data.message) {
          setMessage(data.message);
        }

        // 完了時
        if (data.status === "completed" && data.video_id) {
          eventSource.close();
          setIsConnected(false);
        }
        // エラー時
        if (data.status === "failed") {
          eventSource.close();
          setIsConnected(false);
        }
      } catch (e) {
        // JSON以外のメッセージは無視
      }
    };

    // クリーンアップ
    return () => {
      eventSource.close();
      setIsConnected(false);
    };
  }, [jobId]);

  return {
    status,
    currentStep,
    progress,
    videoId,
    generationId,
    error,
    isConnected,
    message,
  };
}

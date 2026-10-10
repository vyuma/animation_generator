"use client";

import { useEffect, useRef, useCallback } from "react";
import { JobStatus, JobStep } from "@/app/type/apiTypes";
import { getJobStreamUrl } from "@/app/lib/animationApi";

export interface JobStreamResult {
  jobId: string;
  status: JobStatus | null;
  currentStep: JobStep | null;
  progress: number;
  videoId: string | null;
  generationId: number | null;
  error: string | null;
  message: string | null;
}

interface UseMultiJobStreamOptions {
  onProgress?: (result: JobStreamResult) => void;
  onCompleted?: (result: JobStreamResult) => void;
  onError?: (result: JobStreamResult) => void;
}

export function useMultiJobStream(options: UseMultiJobStreamOptions = {}) {
  const eventSourcesRef = useRef<Map<string, EventSource>>(new Map());
  const { onProgress, onCompleted, onError } = options;

  // ジョブの監視を開始
  const startWatching = useCallback((jobId: string) => {
    if (typeof window === "undefined") return;
    if (eventSourcesRef.current.has(jobId)) return; // 既に監視中

    const streamUrl = getJobStreamUrl(jobId);
    const eventSource = new EventSource(streamUrl);

    const createResult = (data: Partial<JobStreamResult>): JobStreamResult => ({
      jobId,
      status: null,
      currentStep: null,
      progress: 0,
      videoId: null,
      generationId: null,
      error: null,
      message: null,
      ...data,
    });

    // 進捗イベント
    eventSource.addEventListener("progress", (event) => {
      try {
        const data = JSON.parse(event.data);
        const result = createResult({
          status: data.status,
          currentStep: data.current_step,
          progress: data.progress <= 1 ? data.progress * 100 : data.progress,
          message: data.message,
        });
        onProgress?.(result);
      } catch (e) {
        console.error("Progress event parse error:", e);
      }
    });

    // 完了イベント
    eventSource.addEventListener("completed", (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("SSE completed event data:", JSON.stringify(data, null, 2));

        // video_idは複数の場所にある可能性がある
        // 優先順位: data.result.video_id > data.video_id > data.data.video_id
        const videoId =
          data.result?.video_id ||
          data.video_id ||
          data.data?.video_id ||
          data.data?.result?.video_id;

        const generationId =
          data.result?.generation_id ||
          data.generation_id ||
          data.data?.generation_id ||
          data.data?.result?.generation_id;

        console.log("Extracted videoId:", videoId);
        console.log("Extracted generationId:", generationId);

        const result = createResult({
          status: "completed",
          videoId,
          generationId,
          progress: 100,
          message: data.message || data.result?.message,
        });
        onCompleted?.(result);
        eventSource.close();
        eventSourcesRef.current.delete(jobId);
      } catch (e) {
        console.error("Completed event parse error:", e);
      }
    });

    // エラーイベント
    eventSource.addEventListener("error", () => {
      if (eventSource.readyState === EventSource.CLOSED) {
        eventSourcesRef.current.delete(jobId);
      }
    });

    // カスタムエラーイベント
    eventSource.addEventListener("job_error", (event) => {
      try {
        const data = JSON.parse(event.data);
        const result = createResult({
          status: "failed",
          error: data.error || "ジョブの実行中にエラーが発生しました",
        });
        onError?.(result);
        eventSource.close();
        eventSourcesRef.current.delete(jobId);
      } catch (e) {
        console.error("Error event parse error:", e);
        onError?.(createResult({
          status: "failed",
          error: "不明なエラーが発生しました",
        }));
      }
    });

    // デフォルトメッセージイベント
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("SSE message event data:", JSON.stringify(data, null, 2));

        // video_idは複数の場所にある可能性がある
        // 優先順位: data.result.video_id > data.video_id > data.data.video_id
        const videoId =
          data.result?.video_id ||
          data.video_id ||
          data.data?.video_id ||
          data.data?.result?.video_id;

        const generationId =
          data.result?.generation_id ||
          data.generation_id ||
          data.data?.generation_id ||
          data.data?.result?.generation_id;

        // 完了ステータスの検出も複数パターン対応
        const isCompleted =
          data.status === "completed" ||
          data.data?.status === "completed" ||
          data.result?.ok === true;

        if (isCompleted) {
          console.log("SSE completed via message event:");
          console.log("Extracted videoId:", videoId);
          console.log("Extracted generationId:", generationId);
        }

        if (isCompleted && videoId) {
          const result = createResult({
            status: "completed",
            videoId,
            generationId,
            progress: 100,
            message: data.message || data.result?.message,
          });
          onCompleted?.(result);
          eventSource.close();
          eventSourcesRef.current.delete(jobId);
          return;
        }

        const isFailed =
          data.status === "failed" ||
          data.data?.status === "failed" ||
          data.result?.ok === false;

        if (isFailed) {
          const result = createResult({
            status: "failed",
            error: data.error || data.result?.error || data.data?.error || "ジョブが失敗しました",
          });
          onError?.(result);
          eventSource.close();
          eventSourcesRef.current.delete(jobId);
          return;
        }

        // 進捗更新
        const progress = data.progress ?? data.data?.progress ?? 0;
        const result = createResult({
          status: data.status || data.data?.status,
          currentStep: data.current_step || data.data?.current_step,
          progress: progress <= 1 ? progress * 100 : progress,
          videoId,
          generationId,
          message: data.message || data.data?.message,
        });
        onProgress?.(result);
      } catch (e) {
        // JSON以外のメッセージは無視
      }
    };

    eventSourcesRef.current.set(jobId, eventSource);
  }, [onProgress, onCompleted, onError]);

  // ジョブの監視を停止
  const stopWatching = useCallback((jobId: string) => {
    const eventSource = eventSourcesRef.current.get(jobId);
    if (eventSource) {
      eventSource.close();
      eventSourcesRef.current.delete(jobId);
    }
  }, []);

  // 全ての監視を停止
  const stopAll = useCallback(() => {
    eventSourcesRef.current.forEach((eventSource) => {
      eventSource.close();
    });
    eventSourcesRef.current.clear();
  }, []);

  // クリーンアップ
  useEffect(() => {
    return () => {
      stopAll();
    };
  }, [stopAll]);

  return {
    startWatching,
    stopWatching,
    stopAll,
  };
}

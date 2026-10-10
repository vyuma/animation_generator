import {
  JobCreateRequest,
  JobCreateResponse,
  JobStatusResponse,
  GenerationSessionResponse,
  GenerationPromptsResponse,
  PlanResponse,
  GenerateFromPlanResponse,
} from "@/app/type/apiTypes";

// フロントエンドからはローカルのAPI Routesを使用
// バックエンドURLはサーバーサイドのみで使用

/**
 * フル生成ジョブ作成
 * POST /api/job/animation/full
 */
export const createFullGenerationJob = async (
  content: string,
  additionalInstructions?: string
): Promise<JobCreateResponse> => {
  const response = await fetch("/api/job/full_generation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: content,
      additional_instructions: additionalInstructions,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "生成ジョブの作成に失敗しました");
  }

  return response.json();
};

/**
 * 編集ジョブ作成
 * POST /api/job/animation/edit
 */
export const createEditJob = async (
  input: JobCreateRequest
): Promise<JobCreateResponse> => {
  const response = await fetch("/api/job/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "編集ジョブの作成に失敗しました");
  }

  return response.json();
};

/**
 * ジョブステータス取得（ポーリング用）
 * GET /api/job/{job_id}/status
 */
export const getJobStatus = async (
  jobId: string
): Promise<JobStatusResponse> => {
  const response = await fetch(`/api/job/${jobId}/status`, {
    method: "GET",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "ジョブステータスの取得に失敗しました");
  }

  return response.json();
};

/**
 * ジョブ削除
 * DELETE /api/job/{job_id}
 */
export const deleteJob = async (jobId: string): Promise<void> => {
  const response = await fetch(`/api/job/${jobId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "ジョブの削除に失敗しました");
  }
};

/**
 * 動画取得（Blob URL）
 * GET /api/animation/{video_id}
 */
export const getAnimation = async (videoId: string): Promise<string> => {
  const response = await fetch(`/api/video/${videoId}`, {
    method: "GET",
  });

  if (!response.ok) {
    throw new Error("動画の取得に失敗しました");
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);
};

/**
 * セッション情報取得
 * GET /api/generation/{generation_id}
 */
export const getGenerationSession = async (
  generationId: number
): Promise<GenerationSessionResponse> => {
  const response = await fetch(`/api/generation/${generationId}`, {
    method: "GET",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "セッション情報の取得に失敗しました");
  }

  return response.json();
};

/**
 * プロンプト履歴取得
 * GET /api/generation/{generation_id}/prompts
 */
export const getGenerationPrompts = async (
  generationId: number
): Promise<GenerationPromptsResponse> => {
  const response = await fetch(`/api/generation/${generationId}/prompts`, {
    method: "GET",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "プロンプト履歴の取得に失敗しました");
  }

  return response.json();
};

/**
 * プランニング（計画立案のみ）
 * POST /api/plan_animation
 */
export const createPlan = async (
  content: string,
  additionalInstructions?: string
): Promise<PlanResponse> => {
  const response = await fetch("/api/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: content,
      additional_instructions: additionalInstructions,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "プランニングに失敗しました");
  }

  return response.json();
};

/**
 * プランから動画生成
 * POST /api/animation
 */
export const generateFromPlan = async (
  generationId: number,
  planContent: string,
  enhancePrompt?: string
): Promise<GenerateFromPlanResponse> => {
  const response = await fetch("/api/job/from_plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      generation_id: generationId,
      content: planContent,
      enhance_prompt: enhancePrompt || "",
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "動画生成に失敗しました");
  }

  return response.json();
};

/**
 * ジョブストリームURL取得（ローカルプロキシ経由）
 */
export const getJobStreamUrl = (jobId: string): string => {
  return `/api/job/${jobId}/stream`;
};

/**
 * ポーリングでジョブ完了を待機
 */
export const waitForJobCompletion = async (
  jobId: string,
  onProgress?: (status: JobStatusResponse) => void,
  pollingInterval: number = 2000
): Promise<JobStatusResponse> => {
  while (true) {
    const status = await getJobStatus(jobId);

    if (onProgress) {
      onProgress(status);
    }

    if (status.status === "completed" || status.status === "failed") {
      return status;
    }

    await new Promise((resolve) => setTimeout(resolve, pollingInterval));
  }
};

/**
 * RAG登録レスポンス
 */
export interface RegisterRagResponse {
  ok: boolean;
  video_id: string;
  message?: string;
}

/**
 * RAG用に動画を登録（いいね機能）
 * POST /api/register_rag/{video_id}
 *
 * 処理内容:
 * 1. manim_codeの内容をgemini2.5-flash-liteで要約
 * 2. 要約した内容を埋め込みベクトル化
 * 3. video_idと要約内容とともにRAGに登録
 */
export const registerVideoToRag = async (
  videoId: string
): Promise<RegisterRagResponse> => {
  const response = await fetch(`/api/register_rag/${videoId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || "RAG登録に失敗しました");
  }

  return response.json();
};

// Job関連の型定義
export type JobStatus = "pending" | "running" | "completed" | "failed";
export type JobType = "generate" | "edit" | "full_generation";
export type JobStep =
  | "initializing"
  | "planning"
  | "generating_script"
  | "linting"
  | "security_check"
  | "preflight"
  | "executing"
  | "refining"
  | "finalizing";

// リクエスト型
export interface ConceptInput {
  text: string;
  additional_instructions?: string | null;
}

export interface EditPrompt {
  generation_id: number;
  prior_video_id: string;
  enhance_prompt: string;
}

export interface JobCreateRequest {
  generation_id: number;
  prior_video_id: string;
  enhance_prompt: string;
}

// トークンコスト型
export interface TokenCostResponse {
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  call_count: number;
  input_cost_usd: number;
  output_cost_usd: number;
  total_cost_usd: number;
  total_cost_jpy: number;
}

// ジョブ作成レスポンス
export interface JobCreateResponse {
  job_id: string;
  status: JobStatus;
  message?: string;
}

// ジョブ結果（完了時）
export interface JobResult {
  ok: boolean;
  generation_id: number;
  video_id: string;
  video_path: string;
  prompt_path?: string;
  manim_code_path?: string;
  message?: string;
  token_cost?: TokenCostResponse;
}

// ジョブステータスレスポンス
export interface JobStatusResponse {
  job_id: string;
  job_type: JobType;
  status: JobStatus;
  progress: number;
  current_step: JobStep;
  message?: string;
  result?: JobResult | null;
  error?: string | null;
  created_at?: string;
  updated_at?: string;
}

// セッション（generation）関連の型
export interface VideoRecord {
  video_id: string;
  video_path: string | null;
  edit_count: number;
  generate_time: string | null;
}

export interface GenerationSessionResponse {
  generation_id: number;
  generate_time: string | null;
  videos: VideoRecord[];
  latest_video_id: string | null;
}

export interface PromptRecord {
  trial: number;
  content: string;
  enhance_prompt: string;
}

export interface GenerationPromptsResponse {
  generation_id: number;
  prompts: PromptRecord[];
  latest_prompt: PromptRecord | null;
}

// SSEイベント型
export interface SSEProgressEvent {
  type: "progress";
  data: {
    status: JobStatus;
    current_step: JobStep;
    progress: number;
    message?: string;
  };
}

export interface SSECompletedEvent {
  type: "completed";
  data: {
    video_id: string;
    generation_id: number;
    status: "completed";
  };
}

export interface SSEErrorEvent {
  type: "error";
  data: {
    error: string;
    status: "failed";
  };
}

export type SSEEvent = SSEProgressEvent | SSECompletedEvent | SSEErrorEvent;

// 動画情報型
export interface VideoInfo {
  video_id: string;
  generation_id: number;
  created_at: string;
  script?: string;
}

// チャットメッセージ型
export type ChatMessageType = "user" | "system" | "video" | "progress" | "error" | "plan";

export interface ChatMessage {
  id: string;
  type: ChatMessageType;
  content: string;
  timestamp: Date;
  // progress type用
  status?: JobStatus;
  currentStep?: JobStep;
  progress?: number;
  // video type用
  videoUrl?: string;
  videoId?: string;
  // plan type用
  planContent?: string;
}

// セッション型（generation_id毎の管理）
export interface GenerationSession {
  generationId: number | null; // サーバーから取得するまではnull
  title: string;
  createdAt: Date;
  messages: ChatMessage[];
  videoUrl?: string;
  videoId?: string;
  status: JobStatus | "idle";
  jobId?: string | null; // 現在実行中のジョブID
  // プランニング関連
  phase?: SessionPhase;
  plan?: PlanInfo | null;
}

// 同期版APIレスポンス（参考用）
export interface FullGenerationSyncResponse {
  ok: boolean;
  message: string;
  video_path: string;
  prompt_path: string;
  manim_code_path: string;
  video_id: string;
  generation_id: number;
  token_cost: TokenCostResponse;
}

// プランニングレスポンス
export interface PlanResponse {
  plan: string;
  generation_id: number | null;
}

// セッションのフェーズ
export type SessionPhase =
  | "idle"           // 初期状態
  | "planning"       // プランニング中
  | "plan_review"    // プランレビュー待ち
  | "generating"     // 動画生成中
  | "completed"      // 完了
  | "failed";        // 失敗

// プラン情報
export interface PlanInfo {
  content: string;
  generationId: number;
  isEditing: boolean;
}

// プランから生成のレスポンス（同期API）
export interface GenerateFromPlanResponse {
  ok: boolean;
  video_id?: string;
  generation_id?: number;
  video_path?: string;
  message?: string;
  job_id?: string; // 非同期の場合
}

// セッションの中間状態（localStorage保存用）
export interface SessionIntermediateState {
  generationId: number;
  phase: SessionPhase;
  plan: PlanInfo | null;
  jobId: string | null;
  updatedAt: number; // タイムスタンプ（ms）
}

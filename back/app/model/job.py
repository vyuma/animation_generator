"""
ジョブ管理用のPydanticモデル定義
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """ジョブの状態"""
    PENDING = "pending"      # 待機中
    RUNNING = "running"      # 実行中
    COMPLETED = "completed"  # 完了
    FAILED = "failed"        # 失敗


class JobStep(str, Enum):
    """ジョブの処理ステップ"""
    INITIALIZING = "initializing"           # 初期化中
    PLANNING = "planning"                   # 計画立案中
    GENERATING_SCRIPT = "generating_script" # スクリプト生成中
    LINTING = "linting"                     # Lint チェック中
    SECURITY_CHECK = "security_check"       # セキュリティチェック中
    PREFLIGHT = "preflight"                 # プリフライト実行中
    EXECUTING = "executing"                 # Manim 実行中
    REFINING = "refining"                   # リファイン中
    FINALIZING = "finalizing"               # 完了処理中


class JobType(str, Enum):
    """ジョブの種類"""
    GENERATE = "generate"       # 新規生成
    EDIT = "edit"               # 編集
    FULL_GENERATION = "full_generation"  # フルパイプライン


class JobInfo(BaseModel):
    """ジョブの詳細情報"""
    job_id: str
    job_type: JobType
    status: JobStatus = JobStatus.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_step: JobStep = JobStep.INITIALIZING
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # 入力パラメータ（参照用）
    generation_id: int | None = None
    video_id: str | None = None


class JobCreateRequest(BaseModel):
    """ジョブ作成リクエスト（編集用）"""
    generation_id: int
    prior_video_id: str
    enhance_prompt: str


class JobCreateResponse(BaseModel):
    """ジョブ作成レスポンス"""
    job_id: str
    status: JobStatus
    message: str = "Job created successfully"


class JobStatusResponse(BaseModel):
    """ジョブ状態レスポンス"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress: float
    current_step: JobStep
    message: str
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class SSEEvent(BaseModel):
    """Server-Sent Event データ"""
    event: str  # "progress", "completed", "error"
    data: dict[str, Any]

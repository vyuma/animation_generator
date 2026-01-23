"""
非同期ジョブ管理API

このルーターは、長時間かかる処理を非同期で実行し、
ポーリングまたはSSEで進捗を取得するためのAPIを提供します。

使用パターン:
1. ポーリング方式:
   - POST /api/job/animation/edit でジョブを作成
   - GET /api/job/{job_id}/status でポーリング

2. ストリーミング方式:
   - POST /api/job/animation/edit/stream でSSE接続
"""
import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.model.job import (
    JobCreateRequest,
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    JobStep,
    JobType,
)
from app.model.model import VideoDatabase, get_video_db
from app.service.base_agent import PlanResponse, SuccessResponse
from app.service.fast_ai_agent import ManimFastAnimationService
from app.service.job_manager import job_manager

router = APIRouter(prefix="/api/job", tags=["job"])

# サービスインスタンス
service = ManimFastAnimationService(
    prompt_dir="prompt",
    base_prompt_file_name="fast_ai_prompts"
)


async def _run_edit_job(
    job_id: str,
    generation_id: int,
    prior_video_id: str,
    enhance_prompt: str,
    db: VideoDatabase,
) -> None:
    """
    編集ジョブを実行（非同期版）

    この関数はバックグラウンドで実行され、
    JobManagerを通じて進捗を通知します。
    """
    try:
        # 開始通知
        job_manager.update_progress(
            job_id, 0.1, JobStep.INITIALIZING, "Starting edit job..."
        )

        # 編集処理を実行（非同期）
        # 詳細な進捗はservice.edit内で更新される
        response: SuccessResponse = await service.edit(
            generation_id=generation_id,
            prior_video_id=prior_video_id,
            enhance_prompt=enhance_prompt,
            max_loop=3,
            job_id=job_id,
        )

        # 成功した場合はDBに保存
        if response.ok and response.video_id and response.video_path:
            job_manager.update_progress(
                job_id, 0.9, JobStep.FINALIZING, "Saving to database..."
            )
            db.edit_video(
                prior_video_id=prior_video_id,
                new_video_path=response.video_path,
                new_video_id=response.video_id,
            )

        # 完了
        job_manager.complete_job(job_id, {
            "ok": response.ok,
            "video_id": response.video_id,
            "video_path": response.video_path,
            "prompt_path": response.prompt_path,
            "manim_code_path": response.manim_code_path,
            "message": response.message,
        })

    except Exception as e:
        job_manager.fail_job(job_id, str(e))


async def _run_full_generation_job(
    job_id: str,
    content: str,
    additional_instructions: str,
    db: VideoDatabase,
) -> None:
    """
    フル生成ジョブを実行（非同期版）
    """
    try:
        # 開始通知
        job_manager.update_progress(
            job_id, 0.05, JobStep.INITIALIZING, "Initializing..."
        )

        # DB に生成セッションを登録
        generate_id = db.generate_prompt()

        job_manager.update_progress(
            job_id, 0.1, JobStep.PLANNING, "Creating animation plan..."
        )

        # 計画立案（非同期）
        plan_response: PlanResponse = await service.plan(
            generation_id=generate_id,
            content=content,
            enhance_prompt=additional_instructions,
        )

        # 動画生成（非同期）
        # 詳細な進捗はservice.main内で更新される
        response: SuccessResponse = await service.main(
            generation_id=generate_id,
            content=plan_response.plan,
            enhance_prompt=additional_instructions,
            max_loop=3,
            job_id=job_id,
        )

        # 成功した場合はDBに保存
        if response.ok and response.video_id:
            job_manager.update_progress(
                job_id, 0.95, JobStep.FINALIZING, "Saving to database..."
            )
            db.generate_video(
                generate_id=generate_id,
                video_id=response.video_id,
                video_path=response.video_path,
                prompt_path=response.prompt_path,
                manim_code_path=response.manim_code_path,
            )

        # 完了
        job_manager.complete_job(job_id, {
            "ok": response.ok,
            "generation_id": generate_id,
            "video_id": response.video_id,
            "video_path": response.video_path,
            "prompt_path": response.prompt_path,
            "manim_code_path": response.manim_code_path,
            "message": response.message,
        })

    except Exception as e:
        job_manager.fail_job(job_id, str(e))


# ============================================================
# ポーリング方式のAPI
# ============================================================

@router.post(
    "/animation/edit",
    response_model=JobCreateResponse,
    summary="編集ジョブを作成",
    description="""
    動画編集ジョブを非同期で開始します。

    このエンドポイントは即座にjob_idを返し、バックグラウンドで処理を実行します。
    進捗は `/api/job/{job_id}/status` で確認できます。

    **使用例:**
    ```
    POST /api/job/animation/edit
    {
        "generation_id": 1,
        "prior_video_id": "abc-123",
        "enhance_prompt": "色を赤に変更"
    }
    ```
    """,
)
async def create_edit_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
    db: VideoDatabase = Depends(get_video_db),
) -> JobCreateResponse:
    """編集ジョブを作成し、即座にジョブIDを返す"""
    # ジョブを作成
    job_id = job_manager.create_job(
        job_type=JobType.EDIT,
        generation_id=request.generation_id,
        video_id=request.prior_video_id,
    )

    # バックグラウンドで非同期実行
    background_tasks.add_task(
        _run_edit_job,
        job_id=job_id,
        generation_id=request.generation_id,
        prior_video_id=request.prior_video_id,
        enhance_prompt=request.enhance_prompt,
        db=db,
    )

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Edit job created. Poll /api/job/{job_id}/status for progress.",
    )


@router.post(
    "/animation/full",
    response_model=JobCreateResponse,
    summary="フル生成ジョブを作成",
    description="""
    動画のフル生成ジョブを非同期で開始します（計画立案 + 生成）。

    このエンドポイントは即座にjob_idを返し、バックグラウンドで処理を実行します。
    """,
)
async def create_full_generation_job(
    content: str,
    additional_instructions: str = "",
    background_tasks: BackgroundTasks = None,
    db: VideoDatabase = Depends(get_video_db),
) -> JobCreateResponse:
    """フル生成ジョブを作成"""
    job_id = job_manager.create_job(job_type=JobType.FULL_GENERATION)

    background_tasks.add_task(
        _run_full_generation_job,
        job_id=job_id,
        content=content,
        additional_instructions=additional_instructions,
        db=db,
    )

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Full generation job created.",
    )


@router.get(
    "/{job_id}/status",
    response_model=JobStatusResponse,
    summary="ジョブの状態を取得",
    description="""
    指定されたジョブIDの現在の状態を返します。

    **レスポンス例:**
    ```json
    {
        "job_id": "abc-123",
        "job_type": "edit",
        "status": "running",
        "progress": 0.5,
        "current_step": "executing",
        "message": "Rendering video...",
        "result": null,
        "error": null
    }
    ```

    **status の値:**
    - `pending`: 待機中
    - `running`: 実行中
    - `completed`: 完了（resultに結果が入る）
    - `failed`: 失敗（errorにエラーメッセージが入る）
    """,
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """ジョブの状態を取得"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        message=job.message,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ============================================================
# ストリーミング方式のAPI (SSE)
# ============================================================

@router.post(
    "/animation/edit/stream",
    summary="編集ジョブをストリーミングで実行",
    description="""
    動画編集ジョブを実行し、進捗をServer-Sent Events (SSE)でリアルタイム送信します。

    **イベント形式:**
    ```
    event: progress
    data: {"job_id": "...", "status": "running", "progress": 0.5, ...}

    event: completed
    data: {"job_id": "...", "status": "completed", "result": {...}}

    event: error
    data: {"job_id": "...", "status": "failed", "error": "..."}
    ```

    **クライアント側の使用例 (JavaScript):**
    ```javascript
    const eventSource = new EventSource('/api/job/animation/edit/stream', {
        method: 'POST',
        body: JSON.stringify({...})
    });

    eventSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        console.log(`Progress: ${data.progress * 100}%`);
    });

    eventSource.addEventListener('completed', (e) => {
        const data = JSON.parse(e.data);
        console.log('Done!', data.result);
        eventSource.close();
    });
    ```
    """,
    response_class=EventSourceResponse,
)
async def stream_edit_job(
    request: JobCreateRequest,
    db: VideoDatabase = Depends(get_video_db),
):
    """編集ジョブをストリーミングで実行"""
    # ジョブを作成
    job_id = job_manager.create_job(
        job_type=JobType.EDIT,
        generation_id=request.generation_id,
        video_id=request.prior_video_id,
    )

    # 別タスクでジョブを実行
    asyncio.create_task(
        _run_edit_job(
            job_id,
            request.generation_id,
            request.prior_video_id,
            request.enhance_prompt,
            db,
        )
    )

    async def event_generator():
        """SSEイベントを生成"""
        async for event in job_manager.stream_job_events(job_id):
            # datetimeをシリアライズ可能な形式に変換
            data = event.data.copy()
            for key in ["created_at", "updated_at"]:
                if key in data and data[key]:
                    if isinstance(data[key], datetime):
                        data[key] = data[key].isoformat()

            yield {
                "event": event.event,
                "data": json.dumps(data, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get(
    "/{job_id}/stream",
    summary="既存ジョブのイベントをストリーミング",
    description="""
    既存のジョブIDを指定して、進捗イベントをSSEで受信します。

    ポーリングでジョブを作成した後、このエンドポイントで
    リアルタイムの進捗を受け取ることができます。
    """,
    response_class=EventSourceResponse,
)
async def stream_job_events(job_id: str):
    """既存ジョブのイベントをストリーミング"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_generator():
        async for event in job_manager.stream_job_events(job_id):
            data = event.data.copy()
            for key in ["created_at", "updated_at"]:
                if key in data and data[key]:
                    if isinstance(data[key], datetime):
                        data[key] = data[key].isoformat()

            yield {
                "event": event.event,
                "data": json.dumps(data, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


# ============================================================
# 管理用API
# ============================================================

@router.delete(
    "/{job_id}",
    summary="ジョブを削除",
    description="指定されたジョブを削除します。実行中のジョブは削除できません。",
)
async def delete_job(job_id: str):
    """ジョブを削除"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running job. Wait for completion or failure.",
        )

    return {"message": f"Job {job_id} deleted"}


@router.post(
    "/cleanup",
    summary="古いジョブをクリーンアップ",
    description="24時間以上前に完了/失敗したジョブを削除します。",
)
async def cleanup_jobs(max_age_hours: int = 24):
    """古いジョブをクリーンアップ"""
    removed = job_manager.cleanup_old_jobs(max_age_hours)
    return {"message": f"Cleaned up {removed} old jobs"}

"""
ジョブ管理サービス
非同期処理のジョブ状態を管理する
"""
import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from threading import Lock
from typing import Any

from loguru import logger

from app.model.job import (
    JobInfo,
    JobStatus,
    JobStep,
    JobType,
    SSEEvent,
)


class JobManager:
    """
    ジョブの作成・状態管理を行うシングルトンクラス

    使用例:
        job_manager = JobManager()
        job_id = job_manager.create_job(JobType.EDIT, generation_id=1)
        job_manager.update_progress(job_id, 0.5, JobStep.EXECUTING, "Rendering...")
        job_manager.complete_job(job_id, result={"video_id": "xxx"})
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._jobs: dict[str, JobInfo] = {}
        self._job_lock = Lock()
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._logger = logger.bind(service="JobManager")
        self._initialized = True

    def create_job(
        self,
        job_type: JobType,
        generation_id: int | None = None,
        video_id: str | None = None,
    ) -> str:
        """
        新しいジョブを作成する

        Args:
            job_type: ジョブの種類
            generation_id: 生成ID
            video_id: 動画ID

        Returns:
            作成されたジョブのID
        """
        job_id = str(uuid.uuid4())

        job_info = JobInfo(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            progress=0.0,
            current_step=JobStep.INITIALIZING,
            message="Job created",
            generation_id=generation_id,
            video_id=video_id,
        )

        with self._job_lock:
            self._jobs[job_id] = job_info
            self._subscribers[job_id] = []

        self._logger.info(f"Job created: {job_id} (type={job_type.value})")
        return job_id

    def get_job(self, job_id: str) -> JobInfo | None:
        """ジョブ情報を取得"""
        with self._job_lock:
            return self._jobs.get(job_id)

    def update_progress(
        self,
        job_id: str,
        progress: float,
        step: JobStep,
        message: str = "",
    ) -> None:
        """
        ジョブの進捗を更新する

        Args:
            job_id: ジョブID
            progress: 進捗率 (0.0 ~ 1.0)
            step: 現在のステップ
            message: メッセージ
        """
        with self._job_lock:
            job = self._jobs.get(job_id)
            if not job:
                self._logger.warning(f"Job not found: {job_id}")
                return

            job.status = JobStatus.RUNNING
            job.progress = min(max(progress, 0.0), 1.0)
            job.current_step = step
            job.message = message
            job.updated_at = datetime.now()

        self._logger.debug(f"Job {job_id}: {step.value} ({progress*100:.0f}%) - {message}")
        self._notify_subscribers(job_id, "progress", job.model_dump())

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        """
        ジョブを完了状態にする

        Args:
            job_id: ジョブID
            result: 結果データ
        """
        with self._job_lock:
            job = self._jobs.get(job_id)
            if not job:
                self._logger.warning(f"Job not found: {job_id}")
                return

            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.current_step = JobStep.FINALIZING
            job.message = "Job completed successfully"
            job.result = result
            job.updated_at = datetime.now()

        self._logger.info(f"Job completed: {job_id}")
        self._notify_subscribers(job_id, "completed", job.model_dump())

    def fail_job(self, job_id: str, error: str) -> None:
        """
        ジョブを失敗状態にする

        Args:
            job_id: ジョブID
            error: エラーメッセージ
        """
        with self._job_lock:
            job = self._jobs.get(job_id)
            if not job:
                self._logger.warning(f"Job not found: {job_id}")
                return

            job.status = JobStatus.FAILED
            job.message = "Job failed"
            job.error = error
            job.updated_at = datetime.now()

        self._logger.error(f"Job failed: {job_id} - {error}")
        self._notify_subscribers(job_id, "error", job.model_dump())

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """
        ジョブの更新を購読する（SSE用）

        Args:
            job_id: ジョブID

        Returns:
            イベントを受け取るキュー
        """
        queue: asyncio.Queue = asyncio.Queue()
        with self._job_lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            self._subscribers[job_id].append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """購読を解除"""
        with self._job_lock:
            if job_id in self._subscribers:
                try:
                    self._subscribers[job_id].remove(queue)
                except ValueError:
                    pass

    def _notify_subscribers(self, job_id: str, event: str, data: dict) -> None:
        """購読者に通知（非同期キューに送信）"""
        # Lock内でsubscribersのコピーを取得し、Lock外でQueue操作を行う
        # これにより、asyncioのイベントループをブロックしない
        with self._job_lock:
            subscribers = list(self._subscribers.get(job_id, []))

        if not subscribers:
            return

        sse_event = SSEEvent(event=event, data=data)
        for queue in subscribers:
            try:
                queue.put_nowait(sse_event)
            except asyncio.QueueFull:
                self._logger.warning(f"Queue full for job {job_id}")

    async def stream_job_events(self, job_id: str) -> AsyncGenerator[SSEEvent, None]:
        """
        ジョブイベントをストリーミングする（SSE用）

        Args:
            job_id: ジョブID

        Yields:
            SSEEvent: イベントデータ
        """
        queue = self.subscribe(job_id)

        try:
            # 現在の状態を即座に送信
            job = self.get_job(job_id)
            if job:
                yield SSEEvent(event="status", data=job.model_dump())

                # 既に完了/失敗している場合は終了
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    return

            # イベントを待機して送信
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event

                    # 完了または失敗で終了
                    if event.event in ("completed", "error"):
                        break
                except asyncio.TimeoutError:
                    # キープアライブ用のping
                    yield SSEEvent(event="ping", data={"timestamp": datetime.now().isoformat()})
        finally:
            self.unsubscribe(job_id, queue)

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        古いジョブを削除

        Args:
            max_age_hours: 保持する最大時間

        Returns:
            削除されたジョブ数
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0

        with self._job_lock:
            to_remove = [
                job_id for job_id, job in self._jobs.items()
                if job.created_at < cutoff and job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            ]
            for job_id in to_remove:
                del self._jobs[job_id]
                if job_id in self._subscribers:
                    del self._subscribers[job_id]
                removed += 1

        if removed > 0:
            self._logger.info(f"Cleaned up {removed} old jobs")

        return removed


# グローバルインスタンス
job_manager = JobManager()

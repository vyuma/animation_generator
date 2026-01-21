# Job API ドキュメント

非同期ジョブ管理APIの仕様書です。長時間かかる動画生成・編集処理を非同期で実行し、Cloudflareのタイムアウト（524エラー）を回避します。

## 概要

```
┌─────────────────────────────────────────────────────────┐
│                   JobManager (共通)                      │
│  - create_job() → job_id                               │
│  - update_status(job_id, status, progress, result)      │
│  - get_status(job_id) → JobStatus                       │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐
│ POST /api/job/  │ │ GET /api/   │ │ POST /api/job/   │
│ animation/edit  │ │ job/{id}/   │ │ animation/edit/  │
│                 │ │ status      │ │ stream           │
│ → job_id       │ │ → JobStatus │ │ → SSE events     │
└─────────────────┘ └─────────────┘ └──────────────────┘
```

## 使用パターン

### パターン1: ポーリング方式

シンプルな実装に適しています。

```javascript
// 1. ジョブを作成
const response = await fetch('/api/job/animation/edit', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    generation_id: 1,
    prior_video_id: 'abc-123',
    enhance_prompt: '色を赤に変更'
  })
});
const { job_id } = await response.json();

// 2. ポーリングで状態を確認
const pollStatus = async () => {
  const statusRes = await fetch(`/api/job/${job_id}/status`);
  const status = await statusRes.json();

  if (status.status === 'completed') {
    console.log('完了:', status.result);
    return status.result;
  } else if (status.status === 'failed') {
    throw new Error(status.error);
  } else {
    console.log(`進捗: ${status.progress * 100}%`);
    setTimeout(pollStatus, 2000); // 2秒後に再確認
  }
};

pollStatus();
```

### パターン2: ストリーミング方式 (SSE)

リアルタイムの進捗表示に適しています。

```javascript
// SSE接続でリアルタイム進捗を取得
const eventSource = new EventSource('/api/job/animation/edit/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    generation_id: 1,
    prior_video_id: 'abc-123',
    enhance_prompt: '色を赤に変更'
  })
});

eventSource.addEventListener('progress', (e) => {
  const data = JSON.parse(e.data);
  console.log(`進捗: ${data.progress * 100}% - ${data.message}`);
  updateProgressBar(data.progress);
});

eventSource.addEventListener('completed', (e) => {
  const data = JSON.parse(e.data);
  console.log('完了:', data.result);
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  console.error('エラー:', data.error);
  eventSource.close();
});
```

---

## API エンドポイント

### POST `/api/job/animation/edit`

動画編集ジョブを非同期で開始します。

#### リクエスト

```json
{
  "generation_id": 1,
  "prior_video_id": "abc-123-def-456",
  "enhance_prompt": "背景色を青に変更してください"
}
```

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `generation_id` | int | ✓ | 生成セッションID |
| `prior_video_id` | string | ✓ | 編集対象の動画ID |
| `enhance_prompt` | string | ✓ | 編集指示 |

#### レスポンス

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Edit job created. Poll /api/job/{job_id}/status for progress."
}
```

---

### GET `/api/job/{job_id}/status`

ジョブの現在の状態を取得します。

#### パスパラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `job_id` | string | ジョブID |

#### レスポンス

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "edit",
  "status": "running",
  "progress": 0.65,
  "current_step": "executing",
  "message": "Rendering video...",
  "result": null,
  "error": null,
  "created_at": "2024-01-21T12:00:00.000Z",
  "updated_at": "2024-01-21T12:01:30.000Z"
}
```

#### ステータス一覧

| status | 説明 |
|--------|------|
| `pending` | 待機中（まだ開始されていない） |
| `running` | 実行中 |
| `completed` | 完了（`result`に結果が入る） |
| `failed` | 失敗（`error`にエラーメッセージが入る） |

#### ステップ一覧

| current_step | 説明 | 目安進捗 |
|--------------|------|---------|
| `initializing` | 初期化中 | 0-10% |
| `planning` | 計画立案中 | 10-20% |
| `generating_script` | スクリプト生成中 | 20-40% |
| `linting` | Lintチェック中 | 40-50% |
| `security_check` | セキュリティチェック中 | 50-55% |
| `preflight` | プリフライト実行中 | 55-70% |
| `executing` | Manim実行中 | 70-90% |
| `refining` | リファイン中 | 90-95% |
| `finalizing` | 完了処理中 | 95-100% |

---

### POST `/api/job/animation/edit/stream`

動画編集ジョブをSSEでストリーミング実行します。

#### リクエスト

`POST /api/job/animation/edit` と同じ

#### レスポンス（SSE形式）

```
event: status
data: {"job_id":"...","status":"pending","progress":0.0,...}

event: progress
data: {"job_id":"...","status":"running","progress":0.3,"current_step":"generating_script",...}

event: progress
data: {"job_id":"...","status":"running","progress":0.7,"current_step":"executing",...}

event: completed
data: {"job_id":"...","status":"completed","progress":1.0,"result":{...}}
```

#### イベントタイプ

| event | 説明 |
|-------|------|
| `status` | 初期状態（接続直後に送信） |
| `progress` | 進捗更新 |
| `completed` | 完了（このイベント後に接続終了） |
| `error` | エラー発生（このイベント後に接続終了） |
| `ping` | キープアライブ（30秒間更新がない場合） |

---

### GET `/api/job/{job_id}/stream`

既存のジョブのイベントをSSEでストリーミングします。

ポーリングでジョブを作成した後、途中からSSEに切り替えたい場合に使用します。

#### パスパラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `job_id` | string | ジョブID |

---

### POST `/api/job/animation/full`

動画のフル生成ジョブを非同期で開始します（計画立案 + 生成）。

#### リクエスト

```json
{
  "content": "ベクトルの加算を説明する動画",
  "additional_instructions": "日本語で、シンプルに"
}
```

---

### DELETE `/api/job/{job_id}`

ジョブを削除します。実行中のジョブは削除できません。

---

### POST `/api/job/cleanup`

24時間以上前に完了/失敗したジョブを削除します。

#### クエリパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `max_age_hours` | int | 24 | 保持する最大時間 |

---

## 完了時のレスポンス例

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "edit",
  "status": "completed",
  "progress": 1.0,
  "current_step": "finalizing",
  "message": "Job completed successfully",
  "result": {
    "ok": true,
    "video_id": "new-video-id-123",
    "video_path": "/videos/new-video-id-123/480p15/GeneratedScene.mp4",
    "prompt_path": "/prompts/1.json",
    "manim_code_path": "/scripts/new-video-id-123.py",
    "message": "Video generated successfully"
  },
  "error": null,
  "created_at": "2024-01-21T12:00:00.000Z",
  "updated_at": "2024-01-21T12:03:45.000Z"
}
```

---

## エラーハンドリング

### 404 Not Found

```json
{
  "detail": "Job not found: invalid-job-id"
}
```

### 400 Bad Request

```json
{
  "detail": "Cannot delete a running job. Wait for completion or failure."
}
```

### 失敗時のレスポンス

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "progress": 0.35,
  "current_step": "generating_script",
  "message": "Job failed",
  "result": null,
  "error": "LLM API rate limit exceeded"
}
```

---

## クライアント実装例

### React Hook

```typescript
import { useState, useEffect, useCallback } from 'react';

type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

interface UseJobOptions {
  pollingInterval?: number;
  useSSE?: boolean;
}

export function useJob(jobId: string | null, options: UseJobOptions = {}) {
  const { pollingInterval = 2000, useSSE = false } = options;
  const [status, setStatus] = useState<JobStatus>('pending');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    if (useSSE) {
      // SSE方式
      const eventSource = new EventSource(`/api/job/${jobId}/stream`);

      const handleEvent = (e: MessageEvent) => {
        const data = JSON.parse(e.data);
        setStatus(data.status);
        setProgress(data.progress);
        if (data.result) setResult(data.result);
        if (data.error) setError(data.error);
      };

      eventSource.addEventListener('progress', handleEvent);
      eventSource.addEventListener('completed', handleEvent);
      eventSource.addEventListener('error', handleEvent);

      return () => eventSource.close();
    } else {
      // ポーリング方式
      const poll = async () => {
        const res = await fetch(`/api/job/${jobId}/status`);
        const data = await res.json();
        setStatus(data.status);
        setProgress(data.progress);
        if (data.result) setResult(data.result);
        if (data.error) setError(data.error);
      };

      const interval = setInterval(poll, pollingInterval);
      poll();

      return () => clearInterval(interval);
    }
  }, [jobId, useSSE, pollingInterval]);

  return { status, progress, result, error };
}
```

### 使用例

```tsx
function VideoEditor() {
  const [jobId, setJobId] = useState<string | null>(null);
  const { status, progress, result, error } = useJob(jobId, { useSSE: true });

  const handleEdit = async () => {
    const res = await fetch('/api/job/animation/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        generation_id: 1,
        prior_video_id: 'video-123',
        enhance_prompt: '色を変更'
      })
    });
    const { job_id } = await res.json();
    setJobId(job_id);
  };

  return (
    <div>
      <button onClick={handleEdit}>編集開始</button>
      {status === 'running' && <ProgressBar value={progress} />}
      {status === 'completed' && <video src={result.video_path} />}
      {status === 'failed' && <ErrorMessage>{error}</ErrorMessage>}
    </div>
  );
}
```

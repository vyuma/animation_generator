# Animation AI Agent API ガイド

このドキュメントでは、動画生成から編集までの一連のフローと、各APIの使用方法、フロントエンドで保持すべき情報について説明します。

---

## 目次

1. [全体フロー概要](#全体フロー概要)
2. [フロントエンドで保持すべき情報](#フロントエンドで保持すべき情報)
3. [API エンドポイント一覧](#api-エンドポイント一覧)
4. [動画生成フロー（同期版）](#動画生成フロー同期版)
5. [動画生成フロー（非同期ジョブ版）](#動画生成フロー非同期ジョブ版)
6. [動画編集フロー](#動画編集フロー)
7. [セッション情報取得](#セッション情報取得)
8. [ジョブステータスの詳細](#ジョブステータスの詳細)
9. [実装例（JavaScript）](#実装例javascript)

---

## 全体フロー概要

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           動画生成・編集フロー                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1. 新規動画生成]                                                           │
│                                                                             │
│      ユーザー入力                                                            │
│          │                                                                  │
│          ▼                                                                  │
│      POST /api/job/animation/full  ──────────► job_id を取得               │
│          │                                                                  │
│          ▼                                                                  │
│      GET /api/job/{job_id}/status  ◄─────────► ポーリングで進捗確認          │
│          │                                                                  │
│          ▼ (status: completed)                                              │
│      result から video_id, generation_id を取得                             │
│          │                                                                  │
│          ▼                                                                  │
│      GET /api/animation/{video_id} ──────────► 動画ファイルを取得            │
│                                                                             │
│                                                                             │
│  [2. 動画編集]                                                               │
│                                                                             │
│      編集指示入力 + video_id + generation_id                                 │
│          │                                                                  │
│          ▼                                                                  │
│      POST /api/job/animation/edit  ──────────► job_id を取得               │
│          │                                                                  │
│          ▼                                                                  │
│      GET /api/job/{job_id}/status  ◄─────────► ポーリングで進捗確認          │
│          │                                                                  │
│          ▼ (status: completed)                                              │
│      result から 新しい video_id を取得                                      │
│          │                                                                  │
│          ▼                                                                  │
│      GET /api/animation/{video_id} ──────────► 編集後の動画ファイルを取得     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## フロントエンドで保持すべき情報

### セッション中に保持する情報

| 情報 | 型 | 取得タイミング | 用途 |
|-----|-----|--------------|------|
| `generation_id` | `number` | 動画生成完了時 | 編集リクエストに必要 |
| `video_id` | `string` | 動画生成/編集完了時 | 動画取得・編集に必要 |
| `job_id` | `string` | ジョブ作成時 | 進捗確認に必要 |

### 状態管理の例（React）

```typescript
interface VideoSession {
  // 現在の動画情報
  generationId: number | null;
  videoId: string | null;

  // ジョブ管理
  currentJobId: string | null;
  jobStatus: 'idle' | 'pending' | 'running' | 'completed' | 'failed';
  jobProgress: number;
  jobStep: string;

  // 動画履歴（編集履歴を保持する場合）
  videoHistory: Array<{
    videoId: string;
    createdAt: Date;
  }>;
}
```

---

## API エンドポイント一覧

### 動画生成・編集（非同期ジョブ版）【推奨】

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| `POST` | `/api/job/animation/full` | フル生成ジョブを作成 |
| `POST` | `/api/job/animation/edit` | 編集ジョブを作成 |
| `GET` | `/api/job/{job_id}/status` | ジョブの状態を取得 |
| `GET` | `/api/job/{job_id}/stream` | ジョブの進捗をSSEで受信 |
| `DELETE` | `/api/job/{job_id}` | ジョブを削除 |

### 動画生成・編集（同期版）

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| `POST` | `/api/full_generation_animation` | フル生成（同期） |
| `POST` | `/api/plan_animation` | 計画立案のみ |
| `POST` | `/api/animation` | 動画生成のみ |
| `POST` | `/api/animation/edit` | 動画編集（同期） |

### 動画取得

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| `GET` | `/api/animation/{video_id}` | 動画ファイル（mp4）を取得 |
| `GET` | `/api/animation/get_info/{video_id}` | 動画のメタ情報を取得 |

### セッション情報取得

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| `GET` | `/api/generation/{generation_id}` | セッションの動画履歴を取得 |
| `GET` | `/api/generation/{generation_id}/prompts` | セッションのプロンプト履歴を取得 |

### テンプレート検索（RAG）

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| `POST` | `/api/search_animation` | 類似動画を検索 |
| `POST` | `/api/register_rag/{video_id}` | 動画をRAGに登録 |

---

## 動画生成フロー（同期版）

> **注意**: 同期版は処理完了まで応答が返らないため、タイムアウトのリスクがあります。
> 本番環境では非同期ジョブ版の使用を推奨します。

### 1. POST /api/full_generation_animation

**リクエスト:**
```json
{
  "text": "二次関数 y = x^2 のグラフを描画し、x=2での接線を表示する",
  "additional_instructions": "日本語で説明を入れてください"
}
```

**レスポンス:**
```json
{
  "ok": true,
  "message": "done",
  "video_path": "abc123-uuid/480p15/GeneratedScene.mp4",
  "prompt_path": "1.json",
  "manim_code_path": "abc123-uuid.py",
  "video_id": "abc123-uuid",
  "generation_id": 1,
  "token_cost": {
    "total_input_tokens": 1500,
    "total_output_tokens": 4000,
    "total_tokens": 5500,
    "call_count": 2,
    "input_cost_usd": 0.003,
    "output_cost_usd": 0.048,
    "total_cost_usd": 0.051,
    "total_cost_jpy": 7.65
  }
}
```

**保持すべき情報:**
- `generation_id`: 編集時に必要
- `video_id`: 動画取得・編集時に必要

---

## 動画生成フロー（非同期ジョブ版）

### Step 1: ジョブを作成

**POST /api/job/animation/full**

```bash
curl -X POST "http://localhost:8000/api/job/animation/full?content=二次関数のグラフを描画&additional_instructions=日本語で"
```

**レスポンス:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Full generation job created."
}
```

**保持すべき情報:** `job_id`

### Step 2: 進捗をポーリング

**GET /api/job/{job_id}/status**

```bash
curl "http://localhost:8000/api/job/550e8400-e29b-41d4-a716-446655440000/status"
```

**レスポンス（実行中）:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "full_generation",
  "status": "running",
  "progress": 0.5,
  "current_step": "linting",
  "message": "Running linter check...",
  "result": null,
  "error": null,
  "created_at": "2026-01-23T10:00:00",
  "updated_at": "2026-01-23T10:01:30"
}
```

**レスポンス（完了）:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "full_generation",
  "status": "completed",
  "progress": 1.0,
  "current_step": "finalizing",
  "message": "Job completed successfully",
  "result": {
    "ok": true,
    "generation_id": 1,
    "video_id": "abc123-uuid",
    "video_path": "abc123-uuid/480p15/GeneratedScene.mp4",
    "prompt_path": "1.json",
    "manim_code_path": "abc123-uuid.py",
    "message": "done"
  },
  "error": null,
  "created_at": "2026-01-23T10:00:00",
  "updated_at": "2026-01-23T10:05:00"
}
```

**保持すべき情報（完了時）:**
- `result.generation_id`
- `result.video_id`

### Step 3: 動画を取得

**GET /api/animation/{video_id}**

```bash
curl "http://localhost:8000/api/animation/abc123-uuid" -o video.mp4
```

---

## 動画編集フロー

### Step 1: 編集ジョブを作成

**POST /api/job/animation/edit**

```json
{
  "generation_id": 1,
  "prior_video_id": "abc123-uuid",
  "enhance_prompt": "グラフの色を青から赤に変更してください"
}
```

**レスポンス:**
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "pending",
  "message": "Edit job created. Poll /api/job/{job_id}/status for progress."
}
```

### Step 2: 進捗をポーリング（生成と同様）

### Step 3: 完了後、新しい video_id を取得

```json
{
  "result": {
    "ok": true,
    "video_id": "def456-uuid",  // ← 新しい video_id
    "video_path": "def456-uuid/480p15/GeneratedScene.mp4",
    ...
  }
}
```

**重要:** 編集後は `video_id` が変わります。次の編集には新しい `video_id` を使用してください。

---

## セッション情報取得

`generation_id` をセッションIDとして使用し、動画の編集履歴やプロンプト履歴を取得できます。

### GET /api/generation/{generation_id}

セッションに紐づく動画の編集履歴を取得します。

**リクエスト:**
```bash
curl "http://localhost:8000/api/generation/1"
```

**レスポンス:**
```json
{
  "generation_id": 1,
  "generate_time": "2026-01-23T10:00:00",
  "videos": [
    {
      "video_id": "abc123-uuid",
      "video_path": "abc123-uuid/480p15/GeneratedScene.mp4",
      "edit_count": 1,
      "generate_time": "2026-01-23T10:05:00"
    },
    {
      "video_id": "def456-uuid",
      "video_path": "def456-uuid/480p15/GeneratedScene.mp4",
      "edit_count": 2,
      "generate_time": "2026-01-23T10:10:00"
    },
    {
      "video_id": "ghi789-uuid",
      "video_path": "ghi789-uuid/480p15/GeneratedScene.mp4",
      "edit_count": 3,
      "generate_time": "2026-01-23T10:15:00"
    }
  ],
  "latest_video_id": "ghi789-uuid"
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `generation_id` | `number` | セッションID |
| `generate_time` | `string \| null` | セッション作成時刻（ISO 8601形式） |
| `videos` | `array` | 動画リスト（`edit_count` 順にソート） |
| `videos[].video_id` | `string` | 動画ID |
| `videos[].video_path` | `string \| null` | 動画ファイルパス |
| `videos[].edit_count` | `number` | 編集回数（1=初回生成、2以降=編集） |
| `videos[].generate_time` | `string \| null` | 動画生成時刻 |
| `latest_video_id` | `string \| null` | 最新の動画ID |

### GET /api/generation/{generation_id}/prompts

セッションに紐づくプロンプトの編集履歴を取得します。

**リクエスト:**
```bash
curl "http://localhost:8000/api/generation/1/prompts"
```

**レスポンス:**
```json
{
  "generation_id": 1,
  "prompts": [
    {
      "trial": 1,
      "content": "二次関数 y = x^2 のグラフを描画し、x=2での接線を表示するアニメーションを作成...",
      "enhance_prompt": "日本語で説明を入れてください"
    },
    {
      "trial": 2,
      "content": "",
      "enhance_prompt": "グラフの色を青から赤に変更してください"
    },
    {
      "trial": 3,
      "content": "",
      "enhance_prompt": "アニメーションの速度を遅くしてください"
    }
  ],
  "latest_prompt": {
    "trial": 3,
    "content": "",
    "enhance_prompt": "アニメーションの速度を遅くしてください"
  }
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `generation_id` | `number` | セッションID |
| `prompts` | `array` | プロンプト履歴リスト（`trial` 順にソート） |
| `prompts[].trial` | `number` | 試行番号（1=初回、2以降=編集） |
| `prompts[].content` | `string` | プロンプト内容（計画など） |
| `prompts[].enhance_prompt` | `string` | 追加の指示 |
| `latest_prompt` | `object \| null` | 最新のプロンプト |

### 活用例

```javascript
// セッション情報を使って編集履歴を表示
async function showSessionHistory(generationId) {
  // 動画履歴を取得
  const sessionRes = await fetch(`/api/generation/${generationId}`);
  const session = await sessionRes.json();

  // プロンプト履歴を取得
  const promptsRes = await fetch(`/api/generation/${generationId}/prompts`);
  const prompts = await promptsRes.json();

  // 編集履歴を結合して表示
  session.videos.forEach((video, index) => {
    const prompt = prompts.prompts[index];
    console.log(`--- 編集 ${video.edit_count} ---`);
    console.log(`動画ID: ${video.video_id}`);
    console.log(`指示: ${prompt?.enhance_prompt || prompt?.content || '(初回生成)'}`);
    console.log(`生成時刻: ${video.generate_time}`);
  });

  console.log(`\n最新の動画: ${session.latest_video_id}`);
}
```

---

## ジョブステータスの詳細

### status（ジョブの状態）

| 値 | 説明 |
|----|------|
| `pending` | 待機中（処理開始前） |
| `running` | 実行中 |
| `completed` | 完了（`result` に結果が入る） |
| `failed` | 失敗（`error` にエラーメッセージが入る） |

### current_step（処理ステップ）

| 値 | 説明 | 目安の進捗 |
|----|------|-----------|
| `initializing` | 初期化中 | 5-10% |
| `planning` | 計画立案中（LLM呼び出し） | 10-30% |
| `generating_script` | スクリプト生成中（LLM呼び出し） | 35% |
| `linting` | Lintチェック中 | 50% |
| `security_check` | セキュリティチェック中 | 55% |
| `preflight` | プリフライト実行中 | 60% |
| `executing` | Manim実行中（動画レンダリング） | 70% |
| `refining` | エラー修正中（リトライ時） | 45% |
| `finalizing` | 完了処理中（DB保存等） | 95% |

---

## 実装例（JavaScript）

### ポーリング方式

```javascript
class AnimationClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.pollingInterval = 2000; // 2秒
  }

  // 動画生成（フルパイプライン）
  async generateVideo(text, additionalInstructions = '') {
    // 1. ジョブを作成
    const createResponse = await fetch(
      `${this.baseUrl}/api/job/animation/full?content=${encodeURIComponent(text)}&additional_instructions=${encodeURIComponent(additionalInstructions)}`,
      { method: 'POST' }
    );
    const { job_id } = await createResponse.json();

    // 2. 完了までポーリング
    const result = await this.waitForJob(job_id);

    return {
      generationId: result.generation_id,
      videoId: result.video_id,
      videoUrl: `${this.baseUrl}/api/animation/${result.video_id}`,
    };
  }

  // 動画編集
  async editVideo(generationId, priorVideoId, enhancePrompt) {
    // 1. 編集ジョブを作成
    const createResponse = await fetch(`${this.baseUrl}/api/job/animation/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        generation_id: generationId,
        prior_video_id: priorVideoId,
        enhance_prompt: enhancePrompt,
      }),
    });
    const { job_id } = await createResponse.json();

    // 2. 完了までポーリング
    const result = await this.waitForJob(job_id);

    return {
      videoId: result.video_id,  // 新しい video_id
      videoUrl: `${this.baseUrl}/api/animation/${result.video_id}`,
    };
  }

  // ジョブ完了待ち（ポーリング）
  async waitForJob(jobId, onProgress = null) {
    while (true) {
      const response = await fetch(`${this.baseUrl}/api/job/${jobId}/status`);
      const status = await response.json();

      // 進捗コールバック
      if (onProgress) {
        onProgress({
          progress: status.progress,
          step: status.current_step,
          message: status.message,
        });
      }

      // 完了または失敗
      if (status.status === 'completed') {
        return status.result;
      }
      if (status.status === 'failed') {
        throw new Error(status.error);
      }

      // 待機
      await new Promise(resolve => setTimeout(resolve, this.pollingInterval));
    }
  }
}

// 使用例
const client = new AnimationClient();

// 動画生成
const video = await client.generateVideo(
  '二次関数のグラフを描画',
  '日本語で説明を入れてください'
);
console.log('生成完了:', video.videoUrl);

// 動画編集
const editedVideo = await client.editVideo(
  video.generationId,
  video.videoId,
  'グラフの色を赤に変更'
);
console.log('編集完了:', editedVideo.videoUrl);
```

### SSE（Server-Sent Events）方式

```javascript
async function streamEditJob(generationId, priorVideoId, enhancePrompt) {
  const response = await fetch('/api/job/animation/edit/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      generation_id: generationId,
      prior_video_id: priorVideoId,
      enhance_prompt: enhancePrompt,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));

        if (data.status === 'running') {
          console.log(`進捗: ${(data.progress * 100).toFixed(0)}% - ${data.message}`);
        } else if (data.status === 'completed') {
          console.log('完了!', data.result);
          return data.result;
        } else if (data.status === 'failed') {
          throw new Error(data.error);
        }
      }
    }
  }
}
```

---

## エラーハンドリング

### よくあるエラー

| HTTPステータス | エラー | 対処法 |
|--------------|-------|--------|
| 404 | `Job not found` | job_id が無効。新しいジョブを作成 |
| 404 | `Video not found` | video_id が無効。generation_id から再取得 |
| 400 | `Cannot delete a running job` | ジョブ完了を待ってから削除 |
| 500 | `Full generation error` | サーバーログを確認。LLM APIの問題の可能性 |

### リトライ戦略

```javascript
async function withRetry(fn, maxRetries = 3, delay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(r => setTimeout(r, delay * (i + 1)));
    }
  }
}
```

---

## 料金情報（token_cost）

API レスポンスには `token_cost` フィールドが含まれ、LLM の使用料金を確認できます。

```json
{
  "token_cost": {
    "total_input_tokens": 1500,
    "total_output_tokens": 4000,
    "total_tokens": 5500,
    "call_count": 2,
    "input_cost_usd": 0.003,
    "output_cost_usd": 0.048,
    "total_cost_usd": 0.051,
    "total_cost_jpy": 7.65
  }
}
```

| フィールド | 説明 |
|-----------|------|
| `total_input_tokens` | 入力トークン数の合計 |
| `total_output_tokens` | 出力トークン数の合計 |
| `call_count` | LLM 呼び出し回数 |
| `total_cost_usd` | 合計コスト（USD） |
| `total_cost_jpy` | 合計コスト（JPY、レート150円/USD） |

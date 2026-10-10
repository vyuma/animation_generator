# Animation AI Agent

自然言語の説明から Manim アニメーションを生成・編集する AI エージェントと、そのフロントエンド（Physiquest Animation）。

```
front/ (Next.js, :3000) ──API Routes──▶ back/ (FastAPI, :8888) ──▶ LLM (Gemini / Claude)
                                             │
                                             ├─ Qual (Manim linter) で静的解析＋自動修正
                                             └─ Manim 0.20.1 でレンダリング (Postgres に履歴保存)
```

## クイックスタート

1. ルートの `.env` に Gemini の API キーを入れる（`.env.example` と同じ内容で作成済み）

   ```bash
   GEMINI_API_KEY=xxxxxxxx
   ```

2. 起動

   ```bash
   docker compose up --build
   ```

3. ブラウザで http://localhost:3000 を開く（API ドキュメントは http://localhost:8888/docs）

ログイン機能（Firebase）は外しており、チャットのセッション一覧はブラウザ（localStorage）に保存されます。動画・プロンプト履歴はバックエンドの Postgres に残ります。

## LLM プロバイダの切り替え（`.env`）

| `LLM_PROVIDER` | 必要な設定 | 備考 |
| --- | --- | --- |
| `gemini`（既定） | `GEMINI_API_KEY` | 既定モデル `gemini-3-flash-preview` |
| `claude-code` | `CLAUDE_CODE_OAUTH_TOKEN` または `ANTHROPIC_API_KEY` | コンテナ内で Claude Code CLI (`claude -p`) を裏で実行。トークンはホストで `claude setup-token` を実行して取得（Claude サブスクリプションで利用可）。既定モデル `sonnet` / 軽量処理 `haiku` |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude API を直接呼び出し |

用途別のモデルは `LLM_MODEL_PRO`（スクリプト生成・編集）、`LLM_MODEL_FLASH`（初回のエラー修正）、`LLM_MODEL_LITE`（プランニング・要約）で上書きできます。

## Manim linter: Qual

生成されたスクリプトは [Qual](https://github.com/Poietra/qual)（`qual-manim`）で静的解析してからレンダリングします。

- 実レンダリング条件（cairo / 854x480 / 15fps）と同じプロファイルで検査
- `square.shift(...)` → `square.animate.shift(...)` などの機械的な修正は自動適用（`QUAL_FIX_MODE`）
- 残った `error` と、confidence が high 以上の `warning` を LLM の修正ループに渡す
- 修正上限に達しても `warning` だけが残っている場合は、レンダリング可能なので実行に進む
- 存在しないクラス名などの実行時エラーは、後段の `manim --dry_run` プリフライトで検出して修正ループに戻す

Qual の knowledge profile が Manim 0.20 系専用のため、バックエンドの Manim は 0.20.1 に上げています。

## フロントエンドだけローカルで動かす

```bash
cd front
bun install
API_URL=http://localhost:8888 bun run dev   # http://localhost:3000
```

## API

**[docs/API_GUIDE.md](./docs/API_GUIDE.md)** を参照してください（フロー、`generation_id` / `video_id` / `job_id`、ポーリングと SSE）。
承認済みプランからの生成は `POST /api/job/animation/generate`（ジョブ）を使います。

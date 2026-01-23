# Animation AI Agent 

## 実行方法

### Dockerを使用する場合

```bash
docker-compose up --build
```

### AIエージェントの責務範囲

AIエージェントは、ユーザーからのリクエストを受け取り、適切なアニメーション生成の計画を立て、その計画に基づいてアニメーションを生成します。具体的には以下のステップを踏みます。


## APIの使用方法

### Swagger UI（自動生成）
```
http://localhost:8000/docs
```

### 詳細なAPIガイド
動画生成から編集までのフロー、フロントエンドで保持すべき情報については以下を参照してください。

**[docs/API_GUIDE.md](./docs/API_GUIDE.md)**

- 全体フロー概要
- フロントエンドで保持すべき情報（`generation_id`, `video_id`, `job_id`）
- ポーリング方式 vs SSE方式の使い分け
- JavaScript実装例


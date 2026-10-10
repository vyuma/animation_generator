# ============================================
# Backend Dockerfile (Cloudflare Tunnel対応)
# ============================================
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# manim と動画生成に必要なシステムパッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ビルドツール
    build-essential \
    curl \
    ca-certificates \
    git \
    pkg-config \
    cmake \
    wget \
    xz-utils \
    # manim 描画ライブラリ
    libcairo2-dev \
    libgirepository1.0-dev \
    libpango1.0-dev \
    libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*


# TeX Live（必要最小限）
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-full texlive-lang-cjk && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    dvisvgm \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg（apt 版: amd64 / arm64 どちらのホストでもネイティブに動く）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv のインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Claude Code CLI（LLM_PROVIDER=claude-code のときに `claude -p` を裏で実行する）
ENV DISABLE_AUTOUPDATER=1
RUN curl -fsSL https://claude.ai/install.sh | bash && claude --version

# 作業ディレクトリを設定
WORKDIR /workspaces/

# back/ ディレクトリの中身だけをコピー
COPY back/ .

# 依存関係をインストール（Manim linter の qual-manim もここで入る）
RUN uv sync --frozen --no-dev && /workspaces/.venv/bin/qual --version

# 仮想環境の PATH を設定
ENV PATH="/workspaces/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspaces/.venv"


# media ディレクトリを作成
RUN mkdir -p /workspaces/media /workspaces/logs /workspaces/script /workspaces/prompts

# ポート設定
EXPOSE 8888

# uvicorn でサーバー起動
# ジョブ状態はプロセス内メモリ (job_manager) で管理するため、ワーカーは 1 つにする
# （複数ワーカーだと /api/job/{id}/status が別プロセスに振られて 404 になる）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "1"]

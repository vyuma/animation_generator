"""
LLM プロバイダの切り替え（.env の LLM_PROVIDER で選択）

  LLM_PROVIDER=gemini       … GEMINI_API_KEY（デフォルト）
  LLM_PROVIDER=anthropic    … ANTHROPIC_API_KEY で Claude API を直接呼ぶ
  LLM_PROVIDER=claude-code  … Claude Code CLI (`claude -p`) を裏で動かす
                               CLAUDE_CODE_OAUTH_TOKEN（`claude setup-token`）または ANTHROPIC_API_KEY

モデルは用途別に LLM_MODEL_PRO / LLM_MODEL_FLASH / LLM_MODEL_LITE で上書きできる。
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

Tier = Literal["pro", "flash", "lite"]

PROVIDER_ALIASES = {
    "gemini": "google",
    "google": "google",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "claude-code": "claude_code",
    "claude_code": "claude_code",
    "openai": "openai",
}

DEFAULT_MODELS: dict[str, dict[Tier, str]] = {
    # コスト削減のため pro も flash を使用（必要なら LLM_MODEL_PRO=gemini-3-pro-preview）
    "google": {"pro": "gemini-3-flash-preview", "flash": "gemini-3-flash-preview", "lite": "gemini-3-flash-preview"},
    "anthropic": {"pro": "claude-sonnet-5-5", "flash": "claude-sonnet-5-5", "lite": "claude-haiku-4-5-20251001"},
    # Claude Code CLI のエイリアス（常に最新世代に解決される）
    "claude_code": {"pro": "sonnet", "flash": "sonnet", "lite": "haiku"},
    "openai": {"pro": "gpt-4o", "flash": "gpt-4o-mini", "lite": "gpt-4o-mini"},
}


def get_provider() -> str:
    raw = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if raw not in PROVIDER_ALIASES:
        raise ValueError(f"Unsupported LLM_PROVIDER: {raw} (gemini | anthropic | claude-code | openai)")
    return PROVIDER_ALIASES[raw]


def get_model_name(tier: Tier, provider: str | None = None) -> str:
    provider = provider or get_provider()
    return os.getenv(f"LLM_MODEL_{tier.upper()}") or DEFAULT_MODELS[provider][tier]


def create_chat_model(tier: Tier, **kwargs) -> BaseChatModel:
    """用途 (tier) に応じた ChatModel を生成する"""
    provider = get_provider()
    model = get_model_name(tier, provider)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your-key":
            raise ValueError("GEMINI_API_KEY is not set. Put your key in the root .env file.")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set (required for LLM_PROVIDER=anthropic).")
        return ChatAnthropic(model=model, api_key=api_key, max_tokens=16000, **kwargs)

    if provider == "claude_code":
        from app.service.llm.claude_code import ChatClaudeCode

        # 認証: Docker 内では CLAUDE_CODE_OAUTH_TOKEN（`claude setup-token`）か ANTHROPIC_API_KEY が必要。
        # ホストで直接動かす場合はログイン済みの Claude Code の認証がそのまま使われる。
        return ChatClaudeCode(model=model)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model_name=model, api_key=os.getenv("OPENAI_API_KEY"), **kwargs)

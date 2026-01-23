import asyncio
import json
import os
import subprocess
import tomllib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langdetect import LangDetectException, detect
from loguru import logger
from pydantic import BaseModel

from app.tools.diff_patcher import DiffPatcher
from app.tools.manim_linter import ManimLinter
from app.tools.secure import is_code_safe

load_dotenv()


# =============================================================================
# トークン料金計算用モック設定
# =============================================================================

@dataclass
class ModelPricing:
    """モデルごとの料金設定（USD/1Mトークン）"""
    input_price_per_million: float  # 入力トークンあたりの料金（USD/1Mトークン）
    output_price_per_million: float  # 出力トークンあたりの料金（USD/1Mトークン）


# モデル料金設定（モック）- Google公式料金を参考に設定
# https://ai.google.dev/pricing で最新料金を確認して更新可能
MODEL_PRICING: dict[str, ModelPricing] = {
    # Google Gemini 3.0
    "gemini-3-pro-preview": ModelPricing(input_price_per_million=2, output_price_per_million=12.0),
    "gemini-3-flash-preview": ModelPricing(input_price_per_million=0.50, output_price_per_million=3.0),
    # Google Gemini 2.x
    "gemini-2.0-flash": ModelPricing(input_price_per_million=0.10, output_price_per_million=0.40),
    "gemini-2.5-flash-preview": ModelPricing(input_price_per_million=0.15, output_price_per_million=0.60),
    # Google Gemini 1.5
    "gemini-1.5-pro": ModelPricing(input_price_per_million=1.25, output_price_per_million=5.0),
    "gemini-1.5-flash": ModelPricing(input_price_per_million=0.075, output_price_per_million=0.30),
    # Anthropic Claude
    "claude-3-5-sonnet-20241022": ModelPricing(input_price_per_million=3.0, output_price_per_million=15.0),
    "claude-3-opus-20240229": ModelPricing(input_price_per_million=15.0, output_price_per_million=75.0),
    "claude-3-haiku-20240307": ModelPricing(input_price_per_million=0.25, output_price_per_million=1.25),
    # OpenAI GPT
    "gpt-4o": ModelPricing(input_price_per_million=2.5, output_price_per_million=10.0),
    "gpt-4o-mini": ModelPricing(input_price_per_million=0.15, output_price_per_million=0.60),
    # デフォルト（不明なモデル用）
    "default": ModelPricing(input_price_per_million=1.0, output_price_per_million=4.0),
}

# USD→JPY 変換レート（モック）- 必要に応じて更新
USD_TO_JPY_RATE: float = 150.0


def get_model_pricing(model_name: str) -> ModelPricing:
    """モデル名から料金設定を取得（部分一致対応）"""
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    # 部分一致で検索
    for key in MODEL_PRICING:
        if key in model_name or model_name in key:
            return MODEL_PRICING[key]
    return MODEL_PRICING["default"]


def calculate_token_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str,
) -> dict[str, float]:
    """
    トークン使用量から料金を計算する（モック関数）

    Args:
        input_tokens: 入力トークン数
        output_tokens: 出力トークン数
        model_name: モデル名

    Returns:
        dict: {
            "input_cost_usd": 入力トークンの料金（USD）,
            "output_cost_usd": 出力トークンの料金（USD）,
            "total_cost_usd": 合計料金（USD）,
            "total_cost_jpy": 合計料金（JPY）,
        }
    """
    pricing = get_model_pricing(model_name)

    input_cost_usd = (input_tokens / 1_000_000) * pricing.input_price_per_million
    output_cost_usd = (output_tokens / 1_000_000) * pricing.output_price_per_million
    total_cost_usd = input_cost_usd + output_cost_usd
    total_cost_jpy = total_cost_usd * USD_TO_JPY_RATE

    return {
        "input_cost_usd": round(input_cost_usd, 6),
        "output_cost_usd": round(output_cost_usd, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "total_cost_jpy": round(total_cost_jpy, 2),
    }


# =============================================================================
# トークン使用量追跡用データ構造
# =============================================================================

@dataclass
class TokenUsage:
    """単一のLLM呼び出しにおけるトークン使用量"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TokenCostSummary:
    """料金サマリー"""
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    total_cost_jpy: float


@dataclass
class TokenUsageSummary:
    """セッション全体のトークン使用量サマリー"""
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    call_count: int
    usages: list[TokenUsage]
    # 料金情報
    cost: TokenCostSummary | None = None


class TokenUsageTracker:
    """トークン使用量を追跡するクラス"""

    def __init__(self, logger):
        self._usages: list[TokenUsage] = []
        self._lock = Lock()
        self._logger = logger

    def add_usage(self, input_tokens: int, output_tokens: int, model_name: str) -> None:
        """トークン使用量を追加し、ログ出力する"""
        total = input_tokens + output_tokens
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            model_name=model_name,
        )
        with self._lock:
            self._usages.append(usage)
        self._logger.info(
            f"[Token Usage] input={input_tokens}, output={output_tokens}, "
            f"total={total} ({model_name})"
        )

    def get_summary(self) -> TokenUsageSummary:
        """現在までのトークン使用量サマリーを取得（料金計算含む）"""
        # Lock内ではデータのコピーのみ行い、料金計算はLock外で実行
        # これによりasyncioイベントループのブロッキングを最小化
        with self._lock:
            usages_copy = list(self._usages)

        # Lock外で集計と料金計算
        total_input = sum(u.input_tokens for u in usages_copy)
        total_output = sum(u.output_tokens for u in usages_copy)

        total_input_cost_usd = 0.0
        total_output_cost_usd = 0.0
        for usage in usages_copy:
            cost = calculate_token_cost(
                usage.input_tokens,
                usage.output_tokens,
                usage.model_name,
            )
            total_input_cost_usd += cost["input_cost_usd"]
            total_output_cost_usd += cost["output_cost_usd"]

        total_cost_usd = total_input_cost_usd + total_output_cost_usd
        total_cost_jpy = total_cost_usd * USD_TO_JPY_RATE

        cost_summary = TokenCostSummary(
            input_cost_usd=round(total_input_cost_usd, 6),
            output_cost_usd=round(total_output_cost_usd, 6),
            total_cost_usd=round(total_cost_usd, 6),
            total_cost_jpy=round(total_cost_jpy, 2),
        )

        return TokenUsageSummary(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            call_count=len(usages_copy),
            usages=usages_copy,
            cost=cost_summary,
        )

    def reset(self) -> TokenUsageSummary:
        """トラッカーをリセットし、リセット前のサマリーを返す"""
        # まずサマリーを取得（Lock内で完結）
        summary = self.get_summary()
        # その後、データをクリア
        with self._lock:
            self._usages.clear()
        return summary

    def log_summary(self) -> None:
        """サマリーをログ出力（料金情報含む）"""
        summary = self.get_summary()
        if summary.call_count > 0:
            cost_info = ""
            if summary.cost:
                cost_info = (
                    f" | Cost: ${summary.cost.total_cost_usd:.4f} USD "
                    f"(¥{summary.cost.total_cost_jpy:.2f} JPY)"
                )
            self._logger.info(
                f"[Token Usage Summary] Total: input={summary.total_input_tokens}, "
                f"output={summary.total_output_tokens}, total={summary.total_tokens} "
                f"({summary.call_count} calls){cost_info}"
            )


class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """LLMのトークン使用量を追跡するコールバックハンドラー"""

    def __init__(self, tracker: TokenUsageTracker):
        super().__init__()
        self.tracker = tracker
        self._current_model: str | None = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        """LLM呼び出し開始時にモデル名を記録"""
        # serializeから モデル名を取得
        if serialized:
            # kwargs にモデル情報がある場合
            model_name = serialized.get("kwargs", {}).get("model", None)
            if model_name:
                self._current_model = model_name
            else:
                # id からモデル名を推測
                id_list = serialized.get("id", [])
                if id_list:
                    self._current_model = id_list[-1]

    def on_llm_end(self, response, **kwargs) -> None:
        """LLM呼び出し終了時にトークン情報を取得してtrackerに追加"""
        model_name = self._current_model or "unknown"

        # responseからトークン情報を取得
        # LLMResult または AIMessage からトークン情報を抽出
        input_tokens = 0
        output_tokens = 0

        # LLMResult の場合
        if hasattr(response, "llm_output") and response.llm_output:
            llm_output = response.llm_output
            # token_usage が直接ある場合 (OpenAI形式)
            if "token_usage" in llm_output:
                token_usage = llm_output["token_usage"]
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
            # usage_metadata がある場合 (Google Gemini形式)
            elif "usage_metadata" in llm_output:
                usage = llm_output["usage_metadata"]
                input_tokens = usage.get("prompt_token_count", 0) or usage.get("input_tokens", 0)
                output_tokens = usage.get("candidates_token_count", 0) or usage.get("output_tokens", 0)

        # generations から取得を試みる (Google Gemini / Anthropic)
        if input_tokens == 0 and output_tokens == 0:
            if hasattr(response, "generations") and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, "message") and gen.message:
                            msg = gen.message
                            # usage_metadata (Google Gemini)
                            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                                usage = msg.usage_metadata
                                input_tokens = getattr(usage, "input_tokens", 0) or usage.get("input_tokens", 0) if isinstance(usage, dict) else getattr(usage, "input_tokens", 0)
                                output_tokens = getattr(usage, "output_tokens", 0) or usage.get("output_tokens", 0) if isinstance(usage, dict) else getattr(usage, "output_tokens", 0)
                            # response_metadata (Anthropic / OpenAI)
                            elif hasattr(msg, "response_metadata") and msg.response_metadata:
                                meta = msg.response_metadata
                                if "usage" in meta:
                                    usage = meta["usage"]
                                    input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                                    output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                                elif "token_usage" in meta:
                                    usage = meta["token_usage"]
                                    input_tokens = usage.get("prompt_tokens", 0)
                                    output_tokens = usage.get("completion_tokens", 0)

        # トークン情報が取得できた場合のみ追加
        if input_tokens > 0 or output_tokens > 0:
            self.tracker.add_usage(input_tokens, output_tokens, model_name)


"""
BaseManimAgentクラス
-----------------------
Manimアニメーションエージェントの基底クラス。
LLMの初期化と共通のユーティリティ関数を提供する関数として定義する。
- LLMの初期化: Google Gemini, Anthropic Claude, OpenAI GPT, xAI grok など複数のモデルプロバイダーに対応。
- 共通ユーティリティ関数: manimの安全性チェック関数やコードフォーマッター、リンター関数など

このクラスはほかのエージェントクラスに継承されて使用される。
特殊メソッド



"""


class TokenCostResponse(BaseModel):
    """API出力用の料金情報"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    total_cost_jpy: float = 0.0


class SuccessResponse(BaseModel):
    ok: bool
    message: str | None = None
    video_path: str | None = None
    prompt_path: str | None = None
    manim_code_path: str | None = None
    video_id: str | None = None
    generation_id: Optional[int] = None
    # 料金情報
    token_cost: TokenCostResponse | None = None


class PlanResponse(BaseModel):
    plan: str
    generation_id: int | None = None


class BaseManimAgent(ABC):
    # 対応言語リスト（新しい言語を追加する場合はここに追加）
    SUPPORTED_LANGUAGES = ["ja", "en"]

    def __init__(self, prompt_dir: str, base_prompt_file_name: str ):
        # Pathの設定
        self.log_path = Path(os.getenv("LOGS_PATH"))
        self.manim_scripts_path = Path(os.getenv("MANIM_SCRIPTS_PATH"))
        self.video_output_path = Path(os.getenv("VIDEO_OUTPUT_PATH"))
        self.user_instruction_path = Path(os.getenv("USER_INSTRUCTION_PATH"))
        # ログのセットアップ
        self.base_logger = self._setup_logger(logger_name=self.__class__.__name__)
        self.diff_patcher = DiffPatcher(self.base_logger)
        # デフォルト言語設定
        self.DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE")
        # プロンプトの読み込み（多言語対応）
        self.prompt_dir = prompt_dir
        self.base_prompt_file_name = base_prompt_file_name
        self.prompts = self._load_all_prompts(prompt_dir=prompt_dir)
        # トークン追跡機能の初期化（LLM初期化より前に必要）
        self._token_tracker = TokenUsageTracker(self.base_logger)
        # LLMの初期化（コールバック付きで返される）
        self.pro_llm = self._load_llm("gemini-3-pro-preview")
        self.flash_llm = self._load_llm("gemini-3-flash-preview")
        self.lite_llm = self._load_llm("gemini-3-flash-preview")
        self.manim_linter = ManimLinter()

        # ローカル関数のpath関連
        self.workspace_path = Path(os.getenv("WORKSPACE_PATH"))

        # pathが存在しない場合には作成する
        for path in [
            self.log_path,
            self.manim_scripts_path,
            self.video_output_path,
            self.user_instruction_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _process_edit_response(self, *, original_script: str, llm_response: str) -> str:
        """
        Delegate to shared diff patcher utility.
        """
        return self.diff_patcher.process_edit_response(original_script=original_script, llm_response=llm_response)

    # クラス変数: 既に追加されたログファイルを追跡（重複防止）
    _registered_log_files: set[str] = set()

    def _setup_logger(self, logger_name: str):
        """
        ログのセットアップ関数
        複数インスタンスが作成されても、同じファイルへのハンドラーは1回のみ追加
        """
        log_file = self.log_path / f"{logger_name}.log"
        log_file_str = str(log_file)

        # 既に追加済みの場合はスキップ（重複防止）
        if log_file_str not in BaseManimAgent._registered_log_files:
            logger.add(log_file, rotation="10 MB", retention="10 days", level="DEBUG")
            BaseManimAgent._registered_log_files.add(log_file_str)

        return logger.bind(name=logger_name)

    def _load_all_prompts(self, prompt_dir: str) -> dict[str, dict]:
        """
        全対応言語のプロンプトを読み込む関数

        ディレクトリ構造:
            prompt/
            ├── ja/
            │   ├── fast_ai_prompts.toml
            │   └── prompts.toml
            └── en/
                ├── fast_ai_prompts.toml
                └── prompts.toml

        Returns:
            {
                "ja": {"chain": {...}, ...},
                "en": {"chain": {...}, ...}
            }
        """
        base_dir = Path(__file__).resolve().parent
        prompts_base_path = base_dir / prompt_dir

        all_prompts: dict[str, dict] = {}

        for lang in self.SUPPORTED_LANGUAGES:
            lang_dir = prompts_base_path / lang
            if not lang_dir.exists():
                self.base_logger.warning(f"Language directory not found: {lang_dir}")
                continue

            all_prompts[lang] = {}
            # 言語ディレクトリ内の全.tomlファイルを読み込む
            for toml_file in lang_dir.glob(f"{self.base_prompt_file_name}.toml"):
                with open(toml_file, "rb") as f:
                    data = tomllib.load(f)
                    # 各tomlファイルの内容をマージ
                    for key, value in data.items():
                        all_prompts[lang][key] = value

            self.base_logger.info(f"Prompts loaded for language: {lang}")

        if not all_prompts:
            raise ValueError(f"No prompts found in {prompts_base_path}")

        return all_prompts

    def _detect_language(self, text: str) -> str:
        """
        テキストから言語を検出する
        対応言語でない場合はデフォルト言語を返す
        """
        try:
            detected = detect(text)
            self.base_logger.debug(f"Detected language: {detected}")
            if detected in self.SUPPORTED_LANGUAGES:
                return detected
            self.base_logger.debug(f"Detected '{detected}' not supported, using default")
            return self.DEFAULT_LANGUAGE
        except LangDetectException:
            self.base_logger.warning("Language detection failed, using default")
            return self.DEFAULT_LANGUAGE

    def get_prompt(self, section: str, key: str, *, user_input: str) -> str:
        """
        ユーザー入力の言語に応じたプロンプトを取得する

        エンジニアはこのメソッドを使うだけで、言語切り替えを意識する必要がない。

        Args:
            section: TOMLのセクション名 (例: "chain")
            key: プロンプトのキー名 (例: "manim_script_generate")
            user_input: ユーザーの入力テキスト（言語検出に使用）

        Returns:
            適切な言語のプロンプト文字列

        Example:
            prompt = self.get_prompt("chain", "manim_script_generate", user_input=content)
        """
        lang = self._detect_language(user_input)

        # フォールバック: 指定言語になければデフォルト言語を使用
        if lang not in self.prompts:
            lang = self.DEFAULT_LANGUAGE

        try:
            return self.prompts[lang][section][key]
        except KeyError:
            # 指定言語にプロンプトがない場合、デフォルト言語で再試行
            self.base_logger.warning(f"Prompt [{section}][{key}] not found for '{lang}', trying default")
            return self.prompts[self.DEFAULT_LANGUAGE][section][key]

    def get_prompt_by_lang(self, section: str, key: str, *, lang: str) -> str:
        """
        指定された言語のプロンプトを取得する

        state に保存された言語を直接使う場合に使用する。

        Args:
            section: TOMLのセクション名 (例: "chain")
            key: プロンプトのキー名 (例: "manim_script_generate")
            lang: 言語コード (例: "ja", "en")

        Returns:
            指定言語のプロンプト文字列

        Example:
            prompt = self.get_prompt_by_lang("chain", "manim_script_generate", lang=state["detected_language"])
        """
        # フォールバック: 指定言語になければデフォルト言語を使用
        if lang not in self.prompts:
            lang = self.DEFAULT_LANGUAGE

        try:
            return self.prompts[lang][section][key]
        except KeyError:
            self.base_logger.warning(f"Prompt [{section}][{key}] not found for '{lang}', trying default")
            return self.prompts[self.DEFAULT_LANGUAGE][section][key]

    def _load_llm(
        self, model_type: str, *, model_provider: str = "google"
    ) -> ChatGoogleGenerativeAI | ChatAnthropic | ChatOpenAI:
        """
        APIによって呼び出す場合のLLMはこの関数の中で定義する。
        例: Google Gemini, Anthropic Claude, OpenAI GPT, xAI grok など

        トークン追跡コールバックがデフォルトで設定される。
        """
        llm: ChatGoogleGenerativeAI | ChatAnthropic | ChatOpenAI | None = None

        if model_provider == "google":
            llm = ChatGoogleGenerativeAI(model=model_type, google_api_key=os.getenv("GEMINI_API_KEY"))
        elif model_provider == "anthropic":
            # Anthropic ClaudeのAPIキーが設定されている場合
            if os.getenv("ANTHROPIC_API_KEY"):
                llm = ChatAnthropic(model=model_type, api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model_provider == "openai":
            llm = ChatOpenAI(model_name=model_type, api_key=os.getenv("OPENAI_API_KEY"))

        if llm is None:
            raise ValueError("Unsupported model provider")

        # トークン追跡コールバックをデフォルトで設定
        callback = TokenTrackingCallbackHandler(self._token_tracker)
        return llm.with_config(callbacks=[callback])

    def _load_local_llm(self, model_type: str) -> ChatOllama:
        # ローカルLLMを作動させる場合の関数
        return ChatOllama(model=model_type)

    # =========================================================================
    # トークン使用量追跡 公開API
    # =========================================================================

    def get_token_usage(self) -> TokenUsageSummary:
        """現在のトークン使用量サマリーを取得"""
        return self._token_tracker.get_summary()

    def reset_token_usage(self) -> TokenUsageSummary:
        """トラッカーをリセットし、リセット前のサマリーを返す"""
        return self._token_tracker.reset()

    def log_token_summary(self) -> None:
        """サマリーをログ出力"""
        self._token_tracker.log_summary()

    def _get_token_cost_response(self) -> TokenCostResponse:
        """トークン使用量サマリーをAPI出力用に変換"""
        summary = self.get_token_usage()
        return TokenCostResponse(
            total_input_tokens=summary.total_input_tokens,
            total_output_tokens=summary.total_output_tokens,
            total_tokens=summary.total_tokens,
            call_count=summary.call_count,
            input_cost_usd=summary.cost.input_cost_usd if summary.cost else 0.0,
            output_cost_usd=summary.cost.output_cost_usd if summary.cost else 0.0,
            total_cost_usd=summary.cost.total_cost_usd if summary.cost else 0.0,
            total_cost_jpy=summary.cost.total_cost_jpy if summary.cost else 0.0,
        )

    def _save_script(self, video_id: str, script: str) -> Path:
        """[Helper] 共通のスクリプト保存処理"""
        tmp_path = self.manim_scripts_path / f"{video_id}.py"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(script)

        self.base_logger.info(f"Script saved to {tmp_path}")
        return tmp_path

    def _save_prompt(self, generation_id: str, content: str, enhance_prompt: str = "") -> str:
        """[Helper] 共通のプロンプト保存処理"""

        prompt_dir = self.user_instruction_path

        # ディレクトリが存在しない場合は作成する

        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)

        prompt_json_path = prompt_dir / f"{generation_id}.json"

        # このファイルが存在しない場合は新規作成する

        if not os.path.isfile(prompt_json_path):
            # jsonファイル作成

            with open(prompt_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "prompt": [
                            {
                                "trial": 1,
                                "content": content,
                                "enhance_prompt": enhance_prompt,
                            }
                        ]
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

                self.base_logger.info(f"Prompt saved to {prompt_json_path}")

        else:
            self.base_logger.info(f"Prompt file already exists: {prompt_json_path}")

            # すでにjsonファイルが存在する場合は追加する

            # まずファイルを読み込む

            with open(prompt_json_path, encoding="utf-8") as f:
                data = json.load(f)

            # データ構造を修正

            if not isinstance(data.get("prompt"), list):
                data["prompt"] = [data["prompt"]] if "prompt" in data else []

            # 新しいプロンプトを追加

            trial_number = len(data["prompt"]) + 1

            data["prompt"].append(
                {
                    "trial": trial_number,
                    "content": content,
                    "enhance_prompt": enhance_prompt,
                }
            )
            # ファイルに書き込む
            with open(prompt_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        return f"{generation_id}.json"

    def _get_script(self, video_id: str) -> str:
        """[Helper] manimスクリプトの取得"""
        script_path = self.manim_scripts_path / f"{video_id}.py"
        with open(script_path, encoding="utf-8") as f:
            script = f.read()
        return script

    def _check_code_security(self, code: str) -> bool:
        """[Helper] manimコードの安全性チェック"""
        return is_code_safe(code)

    def _check_code_lint(self, code: str) -> dict:
        """[Helper] manimコードのリンターチェック"""
        return self.manim_linter.check_code(code)

    def _save_subprocess_logs(self, video_id: str, stdout: str, stderr: str) -> None:
        """[Helper] subprocessのstdout/stderrをvideo_idごとにログファイルに保存"""
        video_log_dir = self.log_path / video_id
        video_log_dir.mkdir(parents=True, exist_ok=True)
        # 新規作成 ファイルが存在しない場合にはファイル作成して保存
        if not os.path.isfile(video_log_dir / "stdout.log") and not os.path.isfile(video_log_dir / "stderr.log"):
            self.base_logger.debug(f"Creating new subprocess log files for video_id: {video_id}")   
            with open(video_log_dir / "stdout.log", "w", encoding="utf-8") as f:
                f.write(stdout or "")
            with open(video_log_dir / "stderr.log", "w", encoding="utf-8") as f:
                f.write(stderr or "")
        else:
            # 追記モードで保存
            self.base_logger.debug(f"Appending subprocess logs to existing files for video_id: {video_id}")
            with open(video_log_dir / "stdout.log", mode = "a", encoding="utf-8") as f:
                f.write("\n\n=== New Execution ===\n\n")
                f.write(stdout or "")
            with open(video_log_dir / "stderr.log", mode = "a", encoding="utf-8") as f:
                f.write("\n\n=== New Execution ===\n\n")
                f.write(stderr or "")

        self.base_logger.debug(f"Subprocess logs saved to {video_log_dir}")

    def _execute_script_low_res(self, script: str, video_id: str) -> str:
        """[Helper] 最低解像度での実行チェック
        副作用: video_idのファイルにスクリプトが保存される
        """
        tmp_path = self._save_script(video_id, script)

        try:
            # dry_runオプションで実行 （実際の動画ファイルは生成しない）
            result = subprocess.run(
                ["manim", "--dry_run", str(tmp_path), "GeneratedScene"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                encoding="utf-8",
            )
            # stdout/stderrをログファイルに保存
            self._save_subprocess_logs(video_id, result.stdout, result.stderr)

            self.base_logger.info(f"Low-res script executed successfully: {tmp_path}")
            return "Success"
        except FileNotFoundError:
            self.base_logger.error(f"File not found: {tmp_path}")
            return "FileNotFoundError"

        except subprocess.CalledProcessError as e:
            # エラー時もstdout/stderrをログファイルに保存
            self._save_subprocess_logs(video_id, e.stdout, e.stderr)

            parsed_error = e.stderr
            # parsed_error = parse_manim_or_python_traceback(e.stderr)
            # parsed_error = format_error_for_llm(parsed_error)
            self.base_logger.error(f"Low-res execution failed: {parsed_error}")
            return parsed_error

    def _execute_script(self, script: str, video_id: str) -> str:
        """[Helper] manimスクリプトの実行
        副作用: video_idのファイルにスクリプトが保存される

        manimスクリプトが正常に実行される。
        """
        script_path = self._save_script(video_id, script)

        try:
            result = subprocess.run(
                [
                    "manim",
                    "--silent",
                    "-v",
                    "error",
                    "--progress_bar",
                    "none",
                    "--media_dir",
                    f"{self.video_output_path}",
                    "--quality",
                    "l",
                    str(script_path),
                    "GeneratedScene",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                encoding="utf-8",
            )

            self.base_logger.info(f"Script executed successfully: {script_path}")
            return "Success"
        except FileNotFoundError:
            self.base_logger.error(f"File not found: {script_path}")
            return "FileNotFoundError"

        except subprocess.CalledProcessError as e:
            # エラー時のみstdout/stderrをログファイルに保存
            self._save_subprocess_logs(video_id, e.stdout, e.stderr)

            parsed_error = e.stderr
            # parsed_error  = parse_manim_or_python_traceback(e.stderr)
            # parsed_error =  format_error_for_llm(parsed_error)
            self.base_logger.error(f"Script execution failed: {parsed_error}")
            return parsed_error

    async def _execute_script_low_res_async(self, script: str, video_id: str) -> str:
        """[Helper] 最低解像度での実行チェック（非同期版）"""
        return await asyncio.to_thread(self._execute_script_low_res, script, video_id)

    async def _execute_script_async(self, script: str, video_id: str) -> str:
        """[Helper] manimスクリプトの実行（非同期版）"""
        return await asyncio.to_thread(self._execute_script, script, video_id)

    @abstractmethod
    async def manim_planner(self, content: str, enhance_prompt: str = "") -> str:
        """
        manimコード生成のための計画立案

        サブクラスで必要ならばオーバーライドして使用する。
        標準実装では flash_llm を使用して計画立案を行う。

        このメソッドでは
        content: 動画生成のためのユーザーの入力
        enhance_prompt: 動画生成を補助するための追加プロンプト

        の二つを受け取り、manimコード生成のための計画を文字列として返す。

        return:str
            manimコード生成のための計画を示す文字列
        """

        pass

    @abstractmethod
    async def generate_video(
        self, video_id: str, content: str, enhance_prompt: str, maxloop: int = 3, job_id: str = ""
    ) -> str:
        """
        サブクラスで実装されるべき抽象的なメソッド（非同期版）

        このコードの中には動画生成のために必要なロジックを実装する。
        このメソッドの中では、
        video_id: 動画の一意な識別子
        content: manimコード生成のための計画立案（planであることに注意する）
        enhance_prompt:動画作成をするための追加プロンプト
        job_id: ジョブID（オプション）。指定すると進捗が更新される

        を受け取る。

        return:
            生成の成功または失敗を示す文字列を返す。
            "Success": 成功
            "bad_request": セキュリティチェックに失敗
            "error": そのほかのエラー
            "failed": その他の失敗
        """
        pass

    @abstractmethod
    async def edit_video(
        self, new_video_id: str, script: str, enhance_prompt: str, max_loop: int = 3, job_id: str = ""
    ) -> str:
        """
        サブクラスで実装されるべき抽象的なメソッド（非同期版）

        このコードの中には動画編集のために必要なロジックを実装する。
        このメソッドの中では、
        script: 既存のmanimスクリプト
        enhance_prompt:動画編集をするための追加プロンプト

        を受け取る。

        return:
            生成の成功または失敗を示す文字列を返す。
            "Success": 成功
            "bad_request": セキュリティチェックに失敗
            "error": そのほかのエラー
            "failed": その他の失敗
        """
        pass

    async def plan(self, generation_id, content: str, enhance_prompt: str = "") -> PlanResponse:
        """
        manimコード生成のための計画立案の共通関数（非同期版）
        """
        try:
            prompt_path = self._save_prompt(generation_id, content, enhance_prompt)
            plan = await self.manim_planner(content, enhance_prompt)
            prompt_path = self._save_prompt(generation_id, plan, enhance_prompt)
            self.base_logger.info(f"Plan generated: {generation_id}")
            self.base_logger.debug(f"Plan details: {prompt_path}")

            return PlanResponse(plan=plan, generation_id=generation_id)

        except Exception as e:
            self.base_logger.error(f"Error in plan: {e}")
            return PlanResponse(plan="", generation_id=None)

    async def main(
        self, generation_id, content: str, enhance_prompt: str, max_loop: int = 3, job_id: str = ""
    ) -> SuccessResponse:
        """
        動画生成のメイン関数（非同期版）

        Args:
            job_id: ジョブID（オプション）。指定すると各処理ステップで進捗が更新される
        """
        # セッション開始時にトークン使用量をリセット
        self.reset_token_usage()

        # video_id(DBに保存するためのpathを一意にするためのID)
        video_id = str(uuid.uuid4())
        self.base_logger.info(f"Starting main video generation for generation_id: {generation_id}")
        self.base_logger.info(f"Video ID: {video_id}")

        # save prompt
        prompt_path = self._save_prompt(generation_id, content, enhance_prompt)

        is_success = await self.generate_video(video_id, content, enhance_prompt, max_loop, job_id)

        # セッション終了時にトークン使用量サマリーをログ出力
        self.log_token_summary()

        # 料金情報を取得
        token_cost = self._get_token_cost_response()

        if is_success == "Success":
            return SuccessResponse(
                ok=True,
                video_id=video_id,
                message="done",
                video_path=str(Path(video_id) / "480p15" / "GeneratedScene.mp4"),
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{video_id}.py"),
                token_cost=token_cost,
            )
        elif is_success == "bad_request":
            return SuccessResponse(
                ok=False,
                message="bad",
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{video_id}.py"),
                token_cost=token_cost,
            )
        else:
            return SuccessResponse(
                ok=False,
                message="failed",
                token_cost=token_cost,
            )

    async def edit(
        self, generation_id: int, prior_video_id: str, enhance_prompt: str, max_loop: int = 3, job_id: str = ""
    ) -> SuccessResponse:
        """
        動画編集の共通関数（非同期版）

        Args:
            job_id: ジョブID（オプション）。指定すると各処理ステップで進捗が更新される
        """
        # セッション開始時にトークン使用量をリセット
        self.reset_token_usage()

        script = self._get_script(prior_video_id)

        new_video_id = str(uuid.uuid4())
        prompt_path = self._save_prompt(generation_id, "", enhance_prompt)
        is_success = await self.edit_video(new_video_id, script, enhance_prompt, max_loop, job_id)

        # セッション終了時にトークン使用量サマリーをログ出力
        self.log_token_summary()

        # 料金情報を取得
        token_cost = self._get_token_cost_response()

        if is_success == "Success":
            return SuccessResponse(
                ok=True,
                message="done",
                video_id=new_video_id,
                video_path=str(Path(new_video_id) / "480p15" / "GeneratedScene.mp4"),
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{new_video_id}.py"),
                token_cost=token_cost,
            )
        elif is_success == "bad_request":
            return SuccessResponse(
                ok=False,
                message="bad",
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{new_video_id}.py"),
                token_cost=token_cost,
            )
        else:
            return SuccessResponse(
                ok=False,
                message="failed",
                token_cost=token_cost,
            )

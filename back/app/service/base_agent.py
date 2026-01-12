import os

import subprocess
from pathlib import Path
from dotenv import load_dotenv
import tomllib
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

import uuid
import json
from pydantic import BaseModel
from typing import Optional
from langdetect import detect, LangDetectException
from loguru import logger

from app.tools.secure import is_code_safe
from app.tools.manim_linter import ManimLinter
from app.tools.manim_lint import parse_manim_or_python_traceback, format_error_for_llm
from app.tools.diff_patcher import DiffPatcher


load_dotenv()


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


class SuccessResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
    video_path: Optional[str] = None
    prompt_path: Optional[str] = None
    manim_code_path: Optional[str] = None
    video_id: Optional[str] = None


class PlanResponse(BaseModel):
    plan: str
    generation_id: Optional[int] = None


class BaseManimAgent(ABC):
    # 対応言語リスト（新しい言語を追加する場合はここに追加）
    SUPPORTED_LANGUAGES = ["ja", "en"]

    def __init__(self, prompt_dir: str = "prompt"):
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
        self.prompts = self._load_all_prompts(prompt_dir=prompt_dir)
        # LLMの初期化
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

    def _setup_logger(self, logger_name: str):
        """
        ログのセットアップ関数
        """
        # log/ ディレクトリが存在しない場合は作成
        log_file = self.log_path / f"{logger_name}.log"
        logger.add(log_file, rotation="10 MB", retention="10 days", level="DEBUG")
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
            for toml_file in lang_dir.glob("*.toml"):
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
        """

        if model_provider == "google":
            return ChatGoogleGenerativeAI(model=model_type, google_api_key=os.getenv("GEMINI_API_KEY"))
        elif model_provider == "anthropic":
            # Anthropic ClaudeのAPIキーが設定されている場合
            if os.getenv("ANTHROPIC_API_KEY"):
                return ChatAnthropic(model=model_type, api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model_provider == "openai":
            return ChatOpenAI(model_name=model_type, api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("Unsupported model provider")

    def _load_local_llm(self, model_type: str) -> ChatOllama:
        # ローカルLLMを作動させる場合の関数
        return ChatOllama(model=model_type)

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

            with open(prompt_json_path, "r", encoding="utf-8") as f:
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
        with open(script_path, "r", encoding="utf-8") as f:
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

        with open(video_log_dir / "stdout.log", "w", encoding="utf-8") as f:
            f.write(stdout or "")
        with open(video_log_dir / "stderr.log", "w", encoding="utf-8") as f:
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
            # stdout/stderrをログファイルに保存
            self._save_subprocess_logs(video_id, result.stdout, result.stderr)

            self.base_logger.info(f"Script executed successfully: {script_path}")
            return "Success"
        except FileNotFoundError:
            self.base_logger.error(f"File not found: {script_path}")
            return "FileNotFoundError"

        except subprocess.CalledProcessError as e:
            # エラー時もstdout/stderrをログファイルに保存
            self._save_subprocess_logs(video_id, e.stdout, e.stderr)

            parsed_error = e.stderr
            # parsed_error  = parse_manim_or_python_traceback(e.stderr)
            # parsed_error =  format_error_for_llm(parsed_error)
            self.base_logger.error(f"Script execution failed: {parsed_error}")
            return parsed_error

    @abstractmethod
    def manim_planner(self, content: str, enhance_prompt: str = "") -> str:
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
    def generate_video(self, video_id: str, content: str, enhance_prompt: str, maxloop: int = 3) -> str:
        """
        サブクラスで実装されるべき抽象的なメソッド

        このコードの中には動画生成のために必要なロジックを実装する。
        このメソッドの中では、
        video_id: 動画の一意な識別子
        content: manimコード生成のための計画立案（planであることに注意する）
        enhance_prompt:動画作成をするための追加プロンプト

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
    def edit_video(self, new_video_id: str, script: str, enhance_prompt: str, max_loop: int = 3) -> str:
        """
        サブクラスで実装されるべき抽象的なメソッド

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

    def plan(self, generation_id, content: str, enhance_prompt: str = "") -> str:
        """
        manimコード生成のための計画立案の共通関数
        """
        try:
            prompt_path = self._save_prompt(generation_id, content, enhance_prompt)
            plan = self.manim_planner(content, enhance_prompt)
            prompt_path: Path = self._save_prompt(generation_id, plan, enhance_prompt)
            self.base_logger.info(f"Plan generated: {generation_id}")
            self.base_logger.debug(f"Plan details: {prompt_path}")

            return PlanResponse(plan=plan, generation_id=generation_id)

        except Exception as e:
            self.base_logger.error(f"Error in plan: {e}")
            return PlanResponse(plan="", generation_id=None)

    def main(self, generation_id, content: str, enhance_prompt: str, max_loop: int = 3) -> SuccessResponse:
        """
        動画生成のメイン関数
        """
        # video_id(DBに保存するためのpathを一意にするためのID)
        video_id = str(uuid.uuid4())

        # save prompt
        prompt_path = self._save_prompt(generation_id, content, enhance_prompt)

        is_success = self.generate_video(video_id, content, enhance_prompt, max_loop)

        if is_success == "Success":
            return SuccessResponse(
                ok=True,
                video_id=video_id,
                message="done",
                video_path=str(Path(video_id) / "480p15" / "GeneratedScene.mp4"),
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{video_id}.py"),
            )
        elif is_success == "bad_request":
            return SuccessResponse(
                ok=False,
                message="bad",
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{video_id}.py"),
            )
        else:
            return SuccessResponse(
                ok=False,
                message="failed",
            )

    def edit(self, generation_id: int, prior_video_id: str, enhance_prompt: str, max_loop: int = 3):
        script = self._get_script(prior_video_id)

        new_video_id = str(uuid.uuid4())
        prompt_path = self._save_prompt(generation_id, "", enhance_prompt)
        is_success = self.edit_video(new_video_id, script, enhance_prompt, max_loop)

        if is_success == "Success":
            return SuccessResponse(
                ok=True,
                message="done",
                video_id=new_video_id,
                video_path=str(Path(new_video_id) / "480p15" / "GeneratedScene.mp4"),
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{new_video_id}.py"),
            )
        elif is_success == "bad_request":
            return SuccessResponse(
                ok=False,
                message="bad",
                prompt_path=str(prompt_path),
                manim_code_path=str(f"{new_video_id}.py"),
            )
        else:
            return SuccessResponse(
                ok=False,
                message="failed",
            )

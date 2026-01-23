# LangGraphのコンポーネント
import asyncio
from typing import Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from app.model.job import JobStep
from app.service.base_agent import BaseManimAgent
from app.service.job_manager import job_manager


class ManimGraphState(TypedDict):
    """
    グラフ全体で引き回す状態。
    """

    # --- 初期入力 ---
    user_request: str  # `content` (構造化説明)
    generation_instructions: str  # `enhance_prompt` (動画の指示)
    animation_plan: str  # 生成されたアニメーションプラン
    video_id: str  # `video_id` (ファイル名用)

    # --- 変化する状態 ---
    current_script: str  # 現在のManimスクリプト (修正対象)
    last_error: str  # 最後に発生したエラーメッセージ (LinterまたはRuntime)
    error_type: Literal["", "lint", "runtime"]  # エラーの種別
    is_bad_request: bool  # 不正リクエストフラグ

    # --- 制御用 ---
    max_retries: int  # 最大試行回数 (元の max_loop)
    current_retry: int  # 現在の試行回数 (元の loop)
    mode: Literal["generate", "edit"]  # 動作モード

    # --- ジョブ管理用（オプション） ---
    job_id: str  # ジョブID（進捗更新用、空文字の場合は更新しない）


class ManimFastAnimationService(BaseManimAgent):
    def __init__(self, prompt_dir: str, base_prompt_file_name: str = "fast_ai_prompts"):
        super().__init__(prompt_dir, base_prompt_file_name=base_prompt_file_name)
        self._current_language: str = self.DEFAULT_LANGUAGE
        # --- LangGraph のグラフを構築 ---
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _update_job_progress(
        self,
        state: ManimGraphState,
        progress: float,
        step: JobStep,
        message: str,
    ) -> None:
        """
        ジョブの進捗を更新するヘルパーメソッド
        job_idが空の場合は何もしない（後方互換性のため）
        """
        job_id = state.get("job_id", "")
        if job_id:
            job_manager.update_progress(job_id, progress, step, message)

    def _get_prompt(self, section: str, key: str) -> str:
        """
        現在の言語設定でプロンプトを取得する
        エンジニアはこれを使うだけでOK
        """
        return self.get_prompt_by_lang(section, key, lang=self._current_language)

    async def _generate_script_with_prompt(self, animation_plan: str) -> str:
        """
        生成済みのアニメーションプランから Manim スクリプトを生成する関数
        """
        prompt_template = self._get_prompt("chain", "manim_script_generate")
        manim_script_prompt = PromptTemplate(
            input_variables=["instructions"],
            template=prompt_template,
        )
        parser = StrOutputParser()

        # プランを instructions として LLM に渡す
        script_chain = manim_script_prompt | self.pro_llm | parser

        # 非同期でLLMを呼び出し
        script_result = await script_chain.ainvoke({"instructions": animation_plan})

        # LLMが出力するマークダウンを削除
        script_result_cleaned = script_result.strip().replace("```python", "").replace("```", "").strip()

        return script_result_cleaned

    async def _generate_initial_script_edit(self, state: ManimGraphState):
        self.base_logger.info("   [+] Edit mode detected. Applying targeted adjustments.")
        prompt_template = self._get_prompt("chain", "fast_ai_edit_initial")
        edit_prompt = PromptTemplate.from_template(prompt_template)
        parser = StrOutputParser()
        chain = edit_prompt | self.pro_llm | parser

        # 非同期でLLMを呼び出し
        llm_response = await chain.ainvoke(
            {
                "edit_instructions": state["user_request"],
                "original_script": state["current_script"],
            }
        )
        self.base_logger.debug(f"llm_response: {llm_response}")

        updated_script = self._process_edit_response(
            original_script=state["current_script"],
            llm_response=llm_response,
        )

        if updated_script != state["current_script"]:
            self.base_logger.debug(f"   [+] Applied edit diff (length: {len(updated_script)})")
        else:
            self.base_logger.info("   [=] Edit diff produced no changes. Keeping original script.")

        return {
            "current_script": updated_script,
            "current_retry": 0,
            "last_error": "",
            "error_type": "",
        }

    async def _generate_initial_script_generate(self, state: ManimGraphState):
        script = await self._generate_script_with_prompt(state["animation_plan"])

        self.base_logger.debug(f"   [+] Initial script generated (length: {len(script)})")
        return {
            "current_script": script,
            "current_retry": 0,
        }

    async def _generate_initial_script(self, state: ManimGraphState):
        """[Node 1] 最初のスクリプトを生成する"""
        self.base_logger.info("--- 1. [Node] Generating Initial Script ---")
        self._update_job_progress(state, 0.35, JobStep.GENERATING_SCRIPT, "Generating initial script...")

        if state["mode"] == "edit":
            return await self._generate_initial_script_edit(state)
        else:
            return await self._generate_initial_script_generate(state)

    async def _run_linter_check(self, state: ManimGraphState):
        self.base_logger.info("--- 2. [Node] Running Manim Linter ---")
        self._update_job_progress(state, 0.50, JobStep.LINTING, "Running linter check...")
        # Linterは同期的な処理なのでto_threadでラップ
        lint_result = await asyncio.to_thread(self._check_code_lint, state["current_script"])
        status = lint_result.get("status")
        issue_count = lint_result.get("issue_count", len(lint_result.get("issues", [])))

        if status == "pass":
            self.base_logger.debug("   [+] Linter check passed with no warnings.")
            return {"last_error": "", "error_type": ""}

        self.base_logger.warning(f"   [-] Linter detected {issue_count} issue(s). Initiating refinement.")
        for issue in lint_result.get("issues", []):
            self.base_logger.debug(
                f"      -> {issue.get('filename')}:{issue.get('lineno')} [{issue.get('code')}] {issue.get('message')}"
            )
        summary = lint_result.get("summary") or ""
        return {"last_error": summary, "error_type": "lint"}

    async def _check_bad_request(self, state: ManimGraphState):
        self.base_logger.info("--- 3. [Node] Checking for Bad Request ---")
        self._update_job_progress(state, 0.55, JobStep.SECURITY_CHECK, "Running security check...")
        # セキュリティチェックは同期的な処理なのでto_threadでラップ
        is_safe = await asyncio.to_thread(self._check_code_security, state["current_script"])
        self.base_logger.debug(f"   [+] Code security check: {'Passed' if is_safe else 'Failed'}")
        if not is_safe:
            return {"is_bad_request": True}
        return {"is_bad_request": False}

    def _handle_execution_result(self, execution_result: str, *, stage: str):
        if execution_result == "Success":
            self.base_logger.info(f"   [+] {stage} succeeded.")
            return {
                "last_error": "",
                "error_type": "",
            }
        if execution_result == "bad_request":
            self.base_logger.warning(f"   [-] {stage} detected unsafe code.")
            return {
                "last_error": "The provided script contains unsafe code.",
                "error_type": "runtime",
            }
        if execution_result == "FileNotFoundError":
            self.base_logger.error(f"   [-] {stage} failed: Manim executable not found.")
            return {
                "last_error": "Manim executable not found.",
                "error_type": "runtime",
            }

        self.base_logger.error(f"   [-] {stage} failed with errors.")
        return {
            "last_error": execution_result,
            "error_type": "runtime",
        }

    async def _execute_and_handle_errors(self, state: ManimGraphState):
        """[Node 4] スクリプトを実行し、エラーを処理する
        解像度を落とした事前実行 -> 本実行の2段階での実行
        """

        script = state["current_script"]
        video_id = state["video_id"]
        self.base_logger.info("--- 4. [Node] Preflight Execution Check ---")
        self._update_job_progress(state, 0.60, JobStep.PREFLIGHT, "Running preflight check...")

        # 非同期でスクリプト実行
        preflight_execution = await self._execute_script_low_res_async(script, video_id)
        preflight_result = self._handle_execution_result(
            preflight_execution,
            stage="Preflight execution",
        )
        if preflight_result["error_type"]:
            return preflight_result

        self.base_logger.info("--- 4. [Node] Executing Manim ---")
        self._update_job_progress(state, 0.70, JobStep.EXECUTING, "Executing Manim script...")
        runtime_execution = await self._execute_script_async(script, video_id)
        runtime_result = self._handle_execution_result(
            runtime_execution,
            stage="Runtime execution",
        )
        return runtime_result

    async def _refine_script_on_error(self, state: ManimGraphState):
        """[Node 5] エラーに基づきスクリプトを修正"""
        self.base_logger.info(f"--- 5. [Node] Refining Script (Attempt {state['current_retry'] + 1}) ---")
        retry_num = state["current_retry"] + 1
        self._update_job_progress(
            state, 0.45, JobStep.REFINING, f"Refining script (attempt {retry_num})..."
        )

        prompt_template = self._get_prompt("chain", "fast_ai_refine_patch")
        repair_prompt = PromptTemplate.from_template(prompt_template)

        parser = StrOutputParser()

        # エラー処理1回目はflash_llm、それ以降はpro_llmを使用
        if state["current_retry"] == 0:
            self.base_logger.debug("   [+] Using flash_llm for first refinement.")
            chain = repair_prompt | self.flash_llm | parser
        else:
            self.base_logger.debug("   [+] Using pro_llm for subsequent refinements.")
            chain = repair_prompt | self.pro_llm | parser

        # 非同期でLLMを呼び出し
        fixed_script_response = await chain.ainvoke(
            {
                "lint_summary": state["last_error"],
                "original_script": state["current_script"],
            }
        )
        self.base_logger.debug(f"llm_response: {fixed_script_response}")

        fixed_script = self._process_edit_response(
            original_script=state["current_script"],
            llm_response=fixed_script_response,
        )

        if fixed_script != state["current_script"]:
            self.base_logger.debug(f"   [+] Script refined via diff (length: {len(fixed_script)})")
        else:
            self.base_logger.info("   [=] Refinement diff produced no changes. Retaining previous script.")

        return {
            "current_script": fixed_script,
            "current_retry": state["current_retry"] + 1,
            "last_error": "",
            "error_type": "",
        }

    def _after_lint_check(self, state: ManimGraphState):
        """[Conditional Edge] リンターエラーか、リトライ上限か"""
        if state["error_type"] == "lint":
            if state["current_retry"] >= state["max_retries"]:
                self.base_logger.warning("--- [Branch] Max Retries Reached (Lint Error). Ending. ---")
                return "end_with_error"
            self.base_logger.info("--- [Branch] Linter Failed. Proceeding to Refine. ---")
            return "refine"
        self.base_logger.debug("--- [Branch] Linter Passed. Proceeding to Security Check. ---")
        return "check_bad_request"

    def _after_bad_request_check(self, state: ManimGraphState):
        """[Conditional Edge] 不正リクエストか"""
        if state["is_bad_request"]:
            self.base_logger.error("--- [Branch] Bad Request. Ending Graph. ---")
            return "end_with_error"
        self.base_logger.debug("--- [Branch] Secure. Proceeding to Execute. ---")
        return "execute"

    def _after_execution(self, state: ManimGraphState):
        """[Conditional Edge] 実行時エラーか、リトライ上限か"""
        if state["error_type"] == "runtime":
            if state["current_retry"] >= state["max_retries"]:
                self.base_logger.warning("--- [Branch] Max Retries Reached (Runtime Error). Ending. ---")
                return "end_with_error"
            self.base_logger.info("--- [Branch] Runtime Error. Proceeding to Refine. ---")
            return "refine"

        self.base_logger.info("--- [Branch] Execution Succeeded. Ending Graph. ---")
        return "end_with_success"

    def _build_graph(self):
        """LangGraphのワークフローを定義・構築する"""
        workflow = StateGraph(ManimGraphState)
        workflow.add_node("generate_initial", self._generate_initial_script)
        workflow.add_node("lint", self._run_linter_check)
        workflow.add_node("check_bad_request", self._check_bad_request)
        workflow.add_node("execute", self._execute_and_handle_errors)
        workflow.add_node("refine", self._refine_script_on_error)
        workflow.set_entry_point("generate_initial")
        workflow.add_edge("generate_initial", "lint")
        workflow.add_edge("refine", "lint")
        workflow.add_conditional_edges(
            "lint",
            self._after_lint_check,
            {
                "refine": "refine",
                "check_bad_request": "check_bad_request",
                "end_with_error": END,
            },
        )
        workflow.add_conditional_edges(
            "check_bad_request",
            self._after_bad_request_check,
            {"execute": "execute", "end_with_error": END},
        )
        workflow.add_conditional_edges(
            "execute",
            self._after_execution,
            {"refine": "refine", "end_with_success": END, "end_with_error": END},
        )
        return workflow

    # ============================================================
    # ==================   Public APIs (async)   =================
    # ============================================================

    async def generate_video(
        self,
        video_id: str,
        content: str,
        enhance_prompt: str,
        maxloop: int = 3,
        job_id: str = "",
    ) -> str:
        """
        動画生成のメイン関数（非同期版）

        Args:
            job_id: ジョブID（オプション）。指定すると各ノードで進捗が更新される
        """
        # ユーザー入力から言語を検出してインスタンスに保存
        self._current_language = self._detect_language(content)
        self.base_logger.info(f"Detected language: {self._current_language}")

        initial_state: ManimGraphState = {
            "user_request": "",
            "generation_instructions": enhance_prompt,
            "animation_plan": content,
            "video_id": video_id,
            "current_script": "",
            "last_error": "",
            "error_type": "",
            "is_bad_request": False,
            "max_retries": maxloop,
            "current_retry": 0,
            "mode": "generate",
            "job_id": job_id,
        }

        # 非同期でLangGraphを実行
        final_state = await self.app.ainvoke(initial_state)

        if final_state["is_bad_request"]:
            self.base_logger.error("--- Graph Finished: Bad Request ---")
            return "bad_request"

        if final_state["last_error"]:
            self.base_logger.error("--- Graph Finished: Error (Max Retries Reached) ---")
            return "error"

        if not final_state["last_error"] and not final_state["is_bad_request"]:
            self.base_logger.info("--- Graph Finished: Success ---")
            return "Success"

        self.base_logger.critical("--- Graph Finished: Fallback (Unknown State) ---")
        return "fall back"

    async def edit_video(
        self,
        video_id: str,
        original_script: str,
        edit_instructions: str,
        maxloop: int = 3,
        job_id: str = "",
    ) -> str:
        """
        既存のスクリプトを編集して動画を生成するメイン関数（非同期版）

        Args:
            job_id: ジョブID（オプション）。指定すると各ノードで進捗が更新される
        """
        # 編集指示から言語を検出してインスタンスに保存
        self._current_language = self._detect_language(edit_instructions)
        self.base_logger.info(f"Detected language: {self._current_language}")

        initial_state: ManimGraphState = {
            "user_request": edit_instructions,
            "generation_instructions": "",
            "animation_plan": "",
            "video_id": video_id,
            "current_script": original_script,
            "last_error": "",
            "error_type": "",
            "is_bad_request": False,
            "max_retries": maxloop,
            "current_retry": 0,
            "mode": "edit",
            "job_id": job_id,
        }

        # 非同期でLangGraphを実行
        final_state = await self.app.ainvoke(initial_state)

        if final_state["is_bad_request"]:
            self.base_logger.error("--- Graph Finished: Bad Request ---")
            return "bad_request"

        if final_state["last_error"]:
            self.base_logger.error("--- Graph Finished: Error (Max Retries Reached) ---")
            return "error"

        if not final_state["last_error"] and not final_state["is_bad_request"]:
            self.base_logger.info("--- Graph Finished: Success ---")
            return "Success"

        self.base_logger.critical("--- Graph Finished: Fallback (Unknown State) ---")
        return "fall back"

    async def manim_planner(self, content: str, enhance_prompt: str) -> str:
        """
        Manimのアニメーションプランを生成する関数（非同期版）
        ユーザー入力 (content) から言語を自動検出してプロンプトを選択する
        """
        # 言語を検出してインスタンスに保存
        self._current_language = self._detect_language(content)

        prompt_template = self._get_prompt("chain", "manim_planer_with_instruct")
        manim_planer = PromptTemplate(
            input_variables=["user_prompt"],
            optional_variables=["video_enhance_prompt"],
            template=prompt_template,
        )
        parser = StrOutputParser()

        chain = manim_planer | self.lite_llm | parser

        # 非同期でLLMを呼び出し
        output: str = await chain.ainvoke({"user_prompt": content, "video_enhance_prompt": enhance_prompt})
        self.base_logger.info(f"Manim planner output: {output}")
        return output

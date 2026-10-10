"""
Claude Code CLI (`claude -p`) を LangChain の ChatModel として使うアダプタ。

API キーの代わりに Claude サブスクリプション（`claude setup-token` で発行した
CLAUDE_CODE_OAUTH_TOKEN）でも動かせる。ツールはすべて無効化し、
純粋なテキスト生成エンジンとして呼び出す。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Manim (Community Edition) developer and animation planner. "
    "Follow the user's instructions exactly and output only what is requested, "
    "in the exact format requested, without any preamble."
)


class ClaudeCodeError(RuntimeError):
    pass


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)


class ChatClaudeCode(BaseChatModel):
    """`claude -p --output-format json` を呼び出す ChatModel"""

    model: str = "sonnet"
    cli_path: str = "claude"
    timeout_sec: float = 600.0

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    # ------------------------------------------------------------------
    def _build_command(self, messages: list[BaseMessage]) -> tuple[list[str], str]:
        system_parts = [_message_text(m) for m in messages if isinstance(m, SystemMessage)]
        body_parts = [_message_text(m) for m in messages if not isinstance(m, SystemMessage)]
        system_prompt = "\n\n".join(system_parts) or DEFAULT_SYSTEM_PROMPT

        cli = shutil.which(self.cli_path) or self.cli_path
        cmd = [
            cli,
            "-p",
            "--output-format", "json",
            "--model", self.model,
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt", system_prompt,
        ]
        return cmd, "\n\n".join(body_parts)

    @staticmethod
    def _parse_output(stdout: str, stderr: str, returncode: int) -> ChatResult:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCodeError(
                f"claude CLI returned non-JSON output (exit={returncode}): {stdout[:500]} {stderr[:500]}"
            ) from e
        if data.get("is_error") or returncode != 0:
            raise ClaudeCodeError(f"claude CLI error (exit={returncode}): {data.get('result') or stderr[:1000]}")

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0)) + int(
            usage.get("cache_creation_input_tokens", 0)
        )
        output_tokens = int(usage.get("output_tokens", 0))
        model_names = list((data.get("modelUsage") or {}).keys())
        message = AIMessage(
            content=data.get("result", ""),
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata={
                "model_name": model_names[0] if model_names else "claude-code",
                "total_cost_usd": data.get("total_cost_usd"),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        cmd, prompt = self._build_command(messages)
        with tempfile.TemporaryDirectory(prefix="claude_llm_") as cwd:
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=cwd,
                    env=os.environ.copy(),
                    timeout=self.timeout_sec,
                )
            except FileNotFoundError as e:
                raise ClaudeCodeError("claude CLI not found. Install Claude Code in the backend image.") from e
        return self._parse_output(proc.stdout, proc.stderr, proc.returncode)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        cmd, prompt = self._build_command(messages)
        with tempfile.TemporaryDirectory(prefix="claude_llm_") as cwd:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=os.environ.copy(),
                )
            except FileNotFoundError as e:
                raise ClaudeCodeError("claude CLI not found. Install Claude Code in the backend image.") from e
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), self.timeout_sec)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise ClaudeCodeError(f"claude CLI timed out after {self.timeout_sec}s") from None
        return self._parse_output(stdout.decode("utf-8"), stderr.decode("utf-8"), proc.returncode or 0)

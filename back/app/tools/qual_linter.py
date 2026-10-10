"""
Qual (https://github.com/Poietra/qual) を使った Manim 専用リンター

- Manim を import / 実行せずに、Scene.play・Mobject ライフサイクル・updater・
  毎フレームの描画コストを静的解析する Rust 製 CLI (`qual-manim` パッケージ) を呼び出す。
- 生成スクリプトを一時ディレクトリに書き出し、実際のレンダリング条件
  (`manim --quality l` = cairo / 854x480 / 15fps) に合わせたプロファイルで検査する。
- `--fix --unsafe-fixes` で機械的な修正 (例: `square.shift(...)` → `square.animate.shift(...)`) を先に適用し、
  残った診断だけを LLM の修正ループに渡す。unsafe な修正も後段の dry_run / 実レンダリングで検証されるため許容する
  (QUAL_FIX_MODE=safe|unsafe|off で変更可)。

check_code() の戻り値は旧 ManimLinter と互換:
    {
        "status": "pass" | "fail",
        "issue_count": int,          # ブロッキング診断の数
        "issues": [ {filename, lineno, col, code, severity, confidence, message, code_line, blocking} ],
        "summary": str,              # LLM に渡すエラー要約
        "has_errors": bool,          # severity=error の診断があるか（warning のみならレンダリングは可能）
        "fixed_code": str | None,    # 安全な自動修正を適用した結果（変更がなければ None）
    }
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

SCRIPT_NAME = "script.py"

# 実レンダリング (`manim --quality l`) と同じ条件で解析する
QUAL_CONFIG = """\
[tool.qual]
manim-version = "0.20"
knowledge-profile = "upstream_0_20"
select = ["MLC", "MLR", "MLP", "MLD"]
min-confidence = "{min_confidence}"
fail-level = "warning"
default-profile = "render"
respect-manim-cfg = false

[[tool.qual.profile]]
name = "render"
renderer = "cairo"
platform = "linux"
pixel-width = 854
pixel-height = 480
frame-rate = 15
"""

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2, "certain": 3}


def _find_qual_binary() -> str | None:
    """qual 実行ファイルを探す（環境変数 → PATH → 実行中 venv の bin）"""
    candidates = [
        os.getenv("QUAL_BIN"),
        shutil.which("qual"),
        str(Path(sys.prefix) / "bin" / "qual"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


class QualLinter:
    """
    Qual CLI のラッパー。

    ブロッキング判定:
      - severity=error（confidence が min_confidence 以上）
      - severity=warning かつ confidence が high/certain
      - info は修正対象にしない（ログのみ）
    """

    def __init__(
        self,
        *,
        min_confidence: str | None = None,
        fix_mode: str | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        self.min_confidence = min_confidence or os.getenv("QUAL_MIN_CONFIDENCE", "medium")
        if self.min_confidence not in CONFIDENCE_ORDER:
            raise ValueError(f"Invalid QUAL_MIN_CONFIDENCE: {self.min_confidence}")
        self.fix_mode = (fix_mode or os.getenv("QUAL_FIX_MODE", "unsafe")).lower()
        if self.fix_mode not in ("safe", "unsafe", "off"):
            raise ValueError(f"Invalid QUAL_FIX_MODE: {self.fix_mode}")
        self.timeout_sec = timeout_sec
        self.binary = _find_qual_binary()
        if self.binary is None:
            logger.warning("qual binary not found. Install `qual-manim` (uv add qual-manim). Lint will be skipped.")

    # ------------------------------------------------------------------
    # 判定ロジック
    # ------------------------------------------------------------------
    @staticmethod
    def _is_blocking(diagnostic: dict[str, Any]) -> bool:
        severity = diagnostic.get("severity")
        confidence = CONFIDENCE_ORDER.get(diagnostic.get("confidence", "low"), 0)
        if severity == "error":
            return True
        if severity == "warning":
            return confidence >= CONFIDENCE_ORDER["high"]
        return False

    def _run_qual(self, workdir: Path, *, fix: bool) -> dict[str, Any]:
        cmd = [self.binary, "check", SCRIPT_NAME, "--format", "json", "--no-cache", "--color", "never"]
        if fix:
            cmd.append("--fix")
            if self.fix_mode == "unsafe":
                cmd.append("--unsafe-fixes")
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout_sec,
        )
        # 0 = 閾値未満, 1 = 閾値以上の診断あり, 2 = 設定/入力/内部エラー
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"qual exited with {proc.returncode}: {proc.stderr.strip()[:2000]}")
        return json.loads(proc.stdout)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------
    def check_code(self, code: str, *, filename: str = "<generated>") -> dict[str, Any]:
        if self.binary is None:
            return self._result([], fixed_code=None, note="qual is not installed; lint skipped")

        with tempfile.TemporaryDirectory(prefix="qual_") as tmp:
            workdir = Path(tmp)
            (workdir / "pyproject.toml").write_text(
                QUAL_CONFIG.format(min_confidence=self.min_confidence), encoding="utf-8"
            )
            script_path = workdir / SCRIPT_NAME
            script_path.write_text(code, encoding="utf-8")

            try:
                report = self._run_qual(workdir, fix=self.fix_mode != "off")
                fixed = script_path.read_text(encoding="utf-8")
                if fixed != code:
                    # --fix 実行時の JSON は修正前の診断を含むため、修正後のコードで再検査する
                    report = self._run_qual(workdir, fix=False)
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
                # リンター自体の故障で生成を止めない（後段の dry_run / 実行で検証される）
                logger.error(f"qual failed, skipping lint: {e}")
                return self._result([], fixed_code=None, note=f"qual failed: {e}")

            fixed_code = fixed if fixed != code else None

        source_lines = (fixed_code or code).splitlines()
        issues: list[dict[str, Any]] = []
        for diag in report.get("diagnostics", []):
            start = diag.get("span", {}).get("start", {})
            lineno = int(start.get("line", 0) or 0)
            code_line = source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""
            issues.append(
                {
                    "filename": filename,
                    "lineno": lineno,
                    "col": int(start.get("column", 0) or 0),
                    "code": diag.get("rule_id", ""),
                    "severity": diag.get("severity", ""),
                    "confidence": diag.get("confidence", ""),
                    "message": diag.get("message", ""),
                    "explanation": diag.get("explanation", ""),
                    "code_line": code_line,
                    "blocking": self._is_blocking(diag),
                }
            )
        return self._result(issues, fixed_code=fixed_code)

    @staticmethod
    def _result(
        issues: list[dict[str, Any]],
        *,
        fixed_code: str | None,
        note: str = "",
    ) -> dict[str, Any]:
        blocking = [i for i in issues if i["blocking"]]
        if not blocking:
            return {
                "status": "pass",
                "issue_count": 0,
                "issues": issues,
                "summary": note,
                "has_errors": False,
                "fixed_code": fixed_code,
            }

        lines: list[str] = []
        for item in blocking:
            detail = f"- line {item['lineno']}:{item['col']} [{item['code']} {item['severity']}] {item['message']}"
            if item["code_line"].strip():
                detail += f"\n    > {item['code_line'].strip()}"
            if item["explanation"]:
                detail += f"\n    hint: {item['explanation']}"
            lines.append(detail)
        summary = "Qual (Manim linter) diagnostics:\n" + "\n".join(lines)
        return {
            "status": "fail",
            "issue_count": len(blocking),
            "issues": issues,
            "summary": summary,
            "has_errors": any(i["severity"] == "error" for i in blocking),
            "fixed_code": fixed_code,
        }

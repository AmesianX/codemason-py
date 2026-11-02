"""PowerShell tool surface + execution substrate (adapted from PowerShellTool.tsx).

Native-Windows tier-2 tool: the model emits PowerShell (or cmd via `cmd /c`),
and this runs it through powershell.exe/pwsh with -EncodedCommand. Gated to
Windows (isPowerShellToolEnabled parity). The security decision lives in the
permission engine (permissions.py); execution here mirrors the Bash tool's
PersistentShell but for PowerShell, with a Get-Location cwd sentinel.
"""

import asyncio
import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from pydantic import Field

from ..base import (
    PermissionResult,
    Tool,
    ToolInputSchema,
    ToolResultSuccess,
    ValidationResult,
    ValidationResultFailure,
    ValidationResultSuccess,
)
from ...utils.platform import is_windows, reap_children_with_this_process
from ...utils.powershell.parser import parse_powershell_command
from ...utils.powershell_detect import get_cached_powershell_path
from ...utils.state import get_cwd, get_original_cwd, is_in_directory, set_cwd as _set_global_cwd
from ...utils.windows_shell import (
    kill_process_tree as _kill_process_tree,
    spawn_group_kwargs as _spawn_group_kwargs,
)
from .command_semantics import interpret_command_result
from .destructive_command_warning import get_destructive_command_warning
from .permissions import powershell_tool_has_permission
from .prompt import get_max_timeout_ms, get_prompt
from .tool_name import POWERSHELL_TOOL_NAME

_DEFAULT_TIMEOUT_MS = 120_000
_MAX_TIMEOUT_MS = 600_000
_MAX_DISPLAY_COMMAND_CHARS = 400


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def is_powershell_tool_enabled() -> bool:
    """Windows-only. Ant defaults on (opt-out via env=0); external opt-in
    (env=1). Mirrors shellToolUtils.isPowerShellToolEnabled."""
    if not is_windows():
        return False
    env = (os.environ.get("CODEMASON_USE_POWERSHELL_TOOL") or "").strip().lower()
    if os.environ.get("USER_TYPE") == "ant":
        return env not in _FALSY  # opt-out
    return env in _TRUTHY  # opt-in


class PowerShellInput(ToolInputSchema):
    """Input schema for the PowerShell tool."""

    command: str = Field(..., description="The PowerShell command to execute")
    timeout: Optional[int] = Field(
        default=None, description=f"Optional timeout in milliseconds (max {_MAX_TIMEOUT_MS})"
    )
    run_in_background: Optional[bool] = Field(
        default=None,
        description="Set to true to run this command in the background. Use Read to read the output later.",
    )
    description: Optional[str] = Field(
        default=None, description="Clear, concise description of what this command does in active voice."
    )
    # NOTE: the TS schema also carries `dangerouslyDisableSandbox`; omitted here
    # because CodeMason has no sandbox substrate (same omission as bash_tool.py).


@dataclass
class PowerShellOutput:
    stdout: str
    stderr: str
    code: int
    interrupted: bool = False
    backgrounded: bool = False


def _to_utf16le_base64(text: str) -> str:
    return base64.b64encode(text.encode("utf-16-le")).decode("ascii")


class PowerShellShell:
    """Per-command PowerShell runner with cwd persistence (Get-Location sentinel)."""

    _instance: Optional["PowerShellShell"] = None

    def __init__(self):
        self._cwd = get_cwd()

    @classmethod
    def get_instance(cls) -> "PowerShellShell":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    async def exec(self, command: str, timeout_ms: int = _DEFAULT_TIMEOUT_MS,
                   abort_event: Optional[asyncio.Event] = None) -> PowerShellOutput:
        pwsh = get_cached_powershell_path()
        if not pwsh:
            return PowerShellOutput(stdout="", stderr="PowerShell is not available on this system.", code=0)

        timeout_sec = timeout_ms / 1000
        pwd_fd, pwd_file = tempfile.mkstemp(prefix="cm-psloc-")
        os.close(pwd_fd)
        # Run the command, capture the effective exit code, persist cwd, re-exit.
        wrapped = (
            f"{command}\n"
            f"$__cmec = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} elseif ($?) {{ 0 }} else {{ 1 }}\n"
            f"(Get-Location).Path | Set-Content -Encoding utf8 -Path {_ps_single_quote(pwd_file)}\n"
            "exit $__cmec\n"
        )
        encoded = _to_utf16le_base64(wrapped)
        args = [pwsh, "-NoProfile", "-NonInteractive", "-NoLogo",
                "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded]

        # Everything spawned from here on dies with this process, however this
        # process dies -- taskkill /F runs no atexit handler, and this shell was
        # outliving the session. Before the spawn on purpose: a job member's
        # children join the job at creation, so there is no window for a
        # grandchild to start outside it. No-op off Windows.
        reap_children_with_this_process()

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env={**os.environ},
            **_spawn_group_kwargs(),
        )


        interrupted = False

        async def _abort_monitor():
            nonlocal interrupted
            if not abort_event:
                return
            await abort_event.wait()
            interrupted = True
            _kill_process_tree(proc)

        abort_task = asyncio.create_task(_abort_monitor()) if abort_event else None
        try:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                _kill_process_tree(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                return PowerShellOutput(
                    stdout="", stderr=f"Command timed out after {timeout_sec}s", code=-1, interrupted=True
                )

            code = proc.returncode
            # Adopt the post-command cwd so Set-Location persists.
            try:
                with open(pwd_file, "r", encoding="utf-8-sig") as f:
                    new_cwd = f.read().strip()
                if new_cwd and os.path.isdir(new_cwd):
                    self._cwd = new_cwd
                    _set_global_cwd(new_cwd)
            except OSError:
                pass

            return PowerShellOutput(
                stdout=(stdout_b or b"").decode("utf-8", "replace"),
                stderr=(stderr_b or b"").decode("utf-8", "replace"),
                code=code if code is not None else 0,
                interrupted=interrupted,
            )
        finally:
            if abort_task and not abort_task.done():
                abort_task.cancel()
                try:
                    await abort_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                os.unlink(pwd_file)
            except OSError:
                pass

    @property
    def cwd(self) -> str:
        return self._cwd


def _ps_single_quote(s: str) -> str:
    """Single-quote a string for PowerShell (double internal single-quotes)."""
    return "'" + s.replace("'", "''") + "'"


class PowerShellTool(Tool):
    """Executes a given PowerShell command and returns its output (native Windows)."""

    name = POWERSHELL_TOOL_NAME
    input_schema = PowerShellInput
    max_result_size_chars = 30_000

    @classmethod
    async def prompt(cls) -> str:
        return get_prompt()

    @classmethod
    async def description(cls, command: str = "", description: str = "", **kwargs) -> str:
        return description or command or "Execute PowerShell command"

    @classmethod
    def is_read_only(cls) -> bool:
        return False

    @classmethod
    async def is_enabled(cls) -> bool:
        return is_powershell_tool_enabled()

    @classmethod
    def needs_permissions(cls) -> bool:
        return True

    @classmethod
    def user_facing_name(cls, input_data: Optional[Dict[str, Any]] = None) -> str:
        return POWERSHELL_TOOL_NAME

    @classmethod
    async def validate_input(cls, input_data: Dict[str, Any], context: Any = None) -> ValidationResult:
        command = input_data.get("command", "") or ""
        if not command.strip():
            return ValidationResultFailure(result=False, message="Command cannot be empty")
        return ValidationResultSuccess()

    @classmethod
    async def check_permissions(cls, input_data: Dict[str, Any], context: Any = None) -> PermissionResult:
        """Parse the command and run the PowerShell permission engine. A 'deny'
        hard-blocks; 'ask' prompts (with the security reason); 'allow' auto-approves
        read-only commands."""
        command = (input_data.get("command", "") or "").strip()
        if not command:
            return PermissionResult(behavior="allow", message="Empty command is safe")
        parsed = parse_powershell_command(command)
        result = powershell_tool_has_permission(command, parsed)
        # Attach a destructive-command note to the message if present.
        warning = get_destructive_command_warning(command)
        if warning and result.behavior != "allow":
            result = PermissionResult(behavior=result.behavior, message=f"{result.message} ({warning})")
        return result

    @classmethod
    def render_tool_use_message(cls, input_data: Dict[str, Any], verbose: bool = False) -> str:
        command = input_data.get("command", "") if isinstance(input_data, dict) else ""
        command = command.strip()
        if len(command) > _MAX_DISPLAY_COMMAND_CHARS:
            command = command[:_MAX_DISPLAY_COMMAND_CHARS] + "..."
        lines = [ln for ln in command.split("\n") if ln.strip()]
        return "\n".join(f"PS> {ln}" if i == 0 else f"    {ln}" for i, ln in enumerate(lines))

    @classmethod
    def render_result_for_assistant(cls, result: PowerShellOutput) -> str:
        error_message = result.stderr.strip()
        if result.interrupted:
            if error_message:
                error_message += "\n"
            error_message += "<error>Command was aborted before completion</error>"
        stdout = result.stdout.strip()
        has_both = stdout and error_message
        return f"{stdout}{chr(10) if has_both else ''}{error_message.strip()}"

    @classmethod
    async def call(cls, input_data: Dict[str, Any], context: Any,
                   can_use_tool: Optional[Callable] = None) -> AsyncGenerator[ToolResultSuccess, None]:
        command = input_data.get("command", "")
        timeout = input_data.get("timeout") or _DEFAULT_TIMEOUT_MS
        if timeout > _MAX_TIMEOUT_MS:
            timeout = _MAX_TIMEOUT_MS

        shell = PowerShellShell.get_instance()

        # run_in_background: spawn detached through the PowerShell substrate,
        # stream output to disk, and return immediately with the task id (same
        # LocalShellTask mechanism the Bash tool uses; kind='powershell' routes
        # the spawn through powershell.exe -EncodedCommand).
        if input_data.get("run_in_background"):
            from ...tasks.local_shell_task import spawn_shell_task
            description = input_data.get("description") or command
            tool_use_id = getattr(context, "tool_use_id", None) or getattr(context, "message_id", None)
            task_id, output_path = spawn_shell_task(
                command, shell.cwd, description=description,
                tool_use_id=tool_use_id, kind="powershell",
            )
            info = (
                f"Command running in background with ID: {task_id}. "
                f"Output is being written to: {output_path}\n"
                f"Use Read to read the output later."
            )
            yield ToolResultSuccess(
                type="result",
                result_for_assistant=info,
                data={"backgroundTaskId": task_id, "output_path": output_path},
            )
            return

        abort_event = getattr(context, "abort_controller", None)
        result = await shell.exec(command, timeout, abort_event=abort_event)

        # Interpret exit code with external-exe semantics (grep/robocopy/etc.).
        is_error, semantic_msg = interpret_command_result(command, result.code, result.stdout, result.stderr)
        if is_error and not result.stderr.strip() and semantic_msg is None:
            result.stderr = (result.stderr + f"\nCommand failed with exit code {result.code}").strip()
        elif not is_error and semantic_msg and not result.stdout.strip():
            result.stdout = semantic_msg

        # Keep the shell cwd inside the session dir (Bash-tool parity).
        if not is_in_directory(shell.cwd, get_original_cwd()):
            shell._cwd = get_original_cwd()
            _set_global_cwd(get_original_cwd())

        yield ToolResultSuccess(
            type="result",
            result_for_assistant=cls.render_result_for_assistant(result),
            data={"stdout": result.stdout, "stderr": result.stderr, "code": result.code},
        )

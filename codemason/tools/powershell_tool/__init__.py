"""PowerShell tool (Tier 2 — native Windows cmd/PowerShell).

Based on Claude Code src/tools/PowerShellTool/. Built bottom-up in phases:
  1. foundation  — tool_name, common_parameters, detection, shell resolution
  2. parser      — utils/powershell parser, static-prefix, dangerous cmdlets
  3. validation  — security, path validation, read-only validation, semantics
  4. permissions — powershell permissions
  5. surface     — the tool class, UI, gating, registration

Activated on native Windows; the model emits PowerShell (or cmd) syntax, so this
is a distinct tool surface from the POSIX-shell Bash tool (Tier 1).
"""

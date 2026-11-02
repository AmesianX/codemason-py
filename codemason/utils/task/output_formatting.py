"""Task output formatting — adapted from src/utils/task/outputFormatting.ts.

Truncates task output for API consumption. When too large, returns the LAST N
chars that fit, prefixed with a header naming the full-output file path.
"""

import os

from .disk_output import get_task_output_path

TASK_MAX_OUTPUT_UPPER_LIMIT = 160_000
TASK_MAX_OUTPUT_DEFAULT = 32_000


def _validate_bounded_int_env_var(value, default: int, upper_limit: int) -> int:
    """Port of validateBoundedIntEnvVar (envValidation.ts): parse a positive
    int from an env value, fall back to `default` on missing/invalid/<=0, and
    cap at `upper_limit`."""
    if not value:
        return default
    try:
        parsed = int(value, 10) if isinstance(value, str) else int(value)
    except (ValueError, TypeError):
        return default
    if parsed <= 0:
        return default
    if parsed > upper_limit:
        return upper_limit
    return parsed


def get_max_task_output_length() -> int:
    """Effective max output length (getMaxTaskOutputLength)."""
    return _validate_bounded_int_env_var(
        os.environ.get("TASK_MAX_OUTPUT_LENGTH"),
        TASK_MAX_OUTPUT_DEFAULT,
        TASK_MAX_OUTPUT_UPPER_LIMIT,
    )


def format_task_output(output: str, task_id: str) -> dict:
    """Truncate output if too large (formatTaskOutput).

    Returns {'content', 'wasTruncated'}. When truncated, the header names the
    full-output file path and the returned content is the last chars that fit.
    """
    max_len = get_max_task_output_length()

    if len(output) <= max_len:
        return {"content": output, "wasTruncated": False}

    file_path = get_task_output_path(task_id)
    header = f"[Truncated. Full output: {file_path}]\n\n"
    # Clamp to >= 0: a very small TASK_MAX_OUTPUT_LENGTH could make this negative,
    # and output[-negative:] would return the FRONT of the output (the source has
    # this same latent slice bug). 0 → just the header. Harmless hardening.
    available_space = max(0, max_len - len(header))
    truncated = output[-available_space:] if available_space else ""

    return {"content": header + truncated, "wasTruncated": True}

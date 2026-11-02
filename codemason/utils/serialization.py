"""
Shared JSON-serialization helper.

`to_jsonable` recursively converts dataclasses / pydantic models / objects
(TextBlock, ToolUseBlock, BetaUsage, APIMessage, uuid.UUID, datetime) into
plain JSON-serializable structures. Used by print_runner (stream-json output)
and session_storage (transcript JSONL) — a raw json.dumps on streamed message
content would raise 'Object of type ToolUseBlock is not JSON serializable'.
"""

from typing import Any


def to_jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {k: to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    model_dump = getattr(v, 'model_dump', None)
    if callable(model_dump):
        try:
            return to_jsonable(model_dump())
        except Exception:
            pass
    if hasattr(v, '__dict__'):
        return {k: to_jsonable(x) for k, x in vars(v).items() if not k.startswith('_')}
    # uuid.UUID / datetime / Path and friends (__slots__, no __dict__) — their
    # str() forms are the canonical JSON representations.
    return str(v)

"""Agent ids — adapted from utils/uuid.ts (createAgentId) and types/ids.ts (isAgentId).

    wm(name?)  ->  `a${name}-${hex16}` when a name is given, else `a${hex16}`
    py4(value) ->  the value if it matches /^a(?:.+-)?[0-9a-f]{16}$/, else None

The id doubles as the sub-agent's task id (the source's `EXK({agentId})` builds
the task with `id: agentId`), and the task id is what names the agent's output
file — so the format is not cosmetic: a value that fails the pattern is not an
agent id and must not be treated as one.
"""

import re
import secrets
from typing import Optional

# /^a(?:.+-)?[0-9a-f]{16}$/
_AGENT_ID_RE = re.compile(r"^a(?:.+-)?[0-9a-f]{16}$")


def create_agent_id(label: Optional[str] = None) -> str:
    """Generate an agent id, optionally carrying a label (createAgentId).

    secrets, not random: the id names a file under a shared output directory, so
    it has to be unguessable for the same reason task ids are.
    """
    suffix = secrets.token_hex(8)
    return f"a{label}-{suffix}" if label else f"a{suffix}"


def is_agent_id(value: object) -> Optional[str]:
    """Return `value` when it is a well-formed agent id, else None (isAgentId)."""
    if not isinstance(value, str):
        return None
    return value if _AGENT_ID_RE.match(value) else None

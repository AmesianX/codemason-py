"""XML tag names for the blocks injected into the conversation —
adapted from src/constants/xml.ts.

These are protocol, not formatting: the task-notification block is parsed back
out in print mode, so a tag that differs between the writer and the reader is a
silently dropped completion. They live in one module for the same reason the
source keeps them in one: three subsystems write these blocks (shell tasks,
agent tasks, the poll pipeline) and they have to agree.
"""

TASK_NOTIFICATION_TAG = "task-notification"
TASK_ID_TAG = "task-id"
TOOL_USE_ID_TAG = "tool-use-id"
TASK_TYPE_TAG = "task-type"
OUTPUT_FILE_TAG = "output-file"
STATUS_TAG = "status"
SUMMARY_TAG = "summary"

# Worktree isolation is not implemented in codemason, so nothing writes these
# yet. Named here anyway: the notification format is shared with the reader, and
# an agent run in a worktree reports where its changes landed.
WORKTREE_TAG = "worktree"
WORKTREE_PATH_TAG = "worktreePath"
WORKTREE_BRANCH_TAG = "worktreeBranch"

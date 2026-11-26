"""Sub-agent requests carry vLLM priority=1; nothing else changes (Rust parity:
services::openai_common::priority_tests).

Run: python3 tests/test_request_priority.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from codemason.services import _openai_common as oc  # noqa: E402


def test_wire_priority_only_for_vllm_subagents_until_refused():
    saved = oc._PRIORITY_REJECTED
    try:
        oc._PRIORITY_REJECTED = False
        assert oc._wire_priority(oc.SUBAGENT_PRIORITY, True) == oc.SUBAGENT_PRIORITY
        assert oc._wire_priority(None, True) is None                    # main: no key
        assert oc._wire_priority(oc.SUBAGENT_PRIORITY, False) is None   # llama.cpp: never
        oc._PRIORITY_REJECTED = True
        assert oc._wire_priority(oc.SUBAGENT_PRIORITY, True) is None    # after a refusal
    finally:
        oc._PRIORITY_REJECTED = saved


if __name__ == '__main__':
    test_wire_priority_only_for_vllm_subagents_until_refused()
    print('PASS test_wire_priority_only_for_vllm_subagents_until_refused')

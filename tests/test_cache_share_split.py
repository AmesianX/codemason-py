"""Public system prefix shared, private part split off (Rust parity:
openai_common.rs system_split_tests + the_public_system_prefix_is_offered...)."""
import sys

from codemason.constants.prompts import PRIVATE_CONTEXT_MARKER as M
from codemason.services import _openai_common as oc

FAILED = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        FAILED.append(name)


out = oc.system_prompt_messages(["intro", "tools", M, "memory", "env"])
check("표지에서 공개/개인 두 메시지로", out == ["intro\n\ntools", "memory\n\nenv"], out)
check("표지는 모델에 안 감", all("\x00" not in m for m in out))
check("표지 없으면 한 메시지", oc.system_prompt_messages(["a", "b"]) == ["a\n\nb"])
check("앞이 비면 분할 안 함", oc.system_prompt_messages([M, "env"]) == ["env"])
check("뒤가 비면 분할 안 함", oc.system_prompt_messages(["intro", M]) == ["intro"])
check("기존 결합 함수도 표지 제거", oc.build_system_prompt_with_cwd_emphasis(["a", M, "b"]) == "a\n\nb")
check("공유 동의는 vLLM 에만",
      oc._wire_vllm_xargs(0.0, True, True) == {"ab_scale": 0, "cache_share_system_prefix": 1})
check("ab 없이도 공유 동의", oc._wire_vllm_xargs(None, True, True) == {"cache_share_system_prefix": 1})
check("llama.cpp 는 never", oc._wire_vllm_xargs(None, False, True) is None)
check("기존 ab 동작 유지", oc._wire_vllm_xargs(1.0, True) == {"ab_scale": 1})

from codemason.utils import cache_session as cs  # noqa: E402
sid = cs.session_id()
check("세션 id 는 게이트웨이 형식", bool(cs._VALID.match(sid)), sid[:6] + "…")
check("세션 id 는 고정", cs.session_id() == sid)

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")

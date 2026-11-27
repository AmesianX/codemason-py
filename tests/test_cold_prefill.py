"""Only COLD prefill is shown (Rust parity: spinner.rs the_prefill_readout_...).

New input -- a WebSearch result, the next question -- is read on every request
and must not light the readout up; already-sent input recomputed is."""
import sys

from codemason.utils import tokens as tk
from codemason.services import _openai_common as oc

FAILED = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        FAILED.append(name)


tk.reset_prefill_state()
tk.set_prefill_expectation(20_000, 0, 80_000)
tk.set_pending_prefill_measured(21_500, 58_500)
check("검색 결과(새 입력)는 콜드 아님", tk.cold_prefill() == (1_500, True), tk.cold_prefill())
tk.reset_prefill_state()
tk.set_prefill_expectation(20_000, 0, 80_000)
tk.set_pending_prefill_measured(58_000, 22_000)
check("새 입력 + 앞 턴 재계산 → 재계산 부분만 콜드", tk.cold_prefill() == (38_000, True), tk.cold_prefill())
tk.reset_prefill_state()
tk.set_prefill_expectation(500, 7_000, 9_000)
check("서버 수치 없으면 클라이언트 추정", tk.cold_prefill() == (7_000, False))

# Per-conversation tracking: a planner request in between is not a break.
oc._prev_by_conv.clear()
main1 = {"messages": [{"role": "system", "content": "S" * 400}, {"role": "user", "content": "Q"}]}
planner = {"messages": [{"role": "system", "content": "planner"}, {"role": "user", "content": "q"}]}
main2 = {"messages": main1["messages"] + [{"role": "assistant", "content": "A"},
                                          {"role": "tool", "content": "R" * 3000}]}
oc._prefill_expectation(main1)
oc._prefill_expectation(planner)
new, cold, total = oc._prefill_expectation(main2)
check("사이에 다른 대화가 끼어도 이어붙임으로 판정", cold == 0 and 0 < new < total, (new, cold, total))
main3 = {"messages": [main1["messages"][0], {"role": "user", "content": "Q-changed"}] + main2["messages"][2:]}
new3, cold3, _ = oc._prefill_expectation(main3)
check("앞 메시지가 바뀌면 클라이언트가 콜드로 셈", cold3 > 0, (new3, cold3))

tk.reset_prefill_state()
print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")

"""한국어 UI 문자열.

키는 영어 원문. 없는 항목은 영어로 표시되므로 부분 번역도 안전하다.
자리표시자는 이름 있는 형식({count})이라 어순을 바꿔도 된다.
"""

MESSAGES = {
    # /remote 상태 화면
    'Remote channel is off  ({reason})': '원격 채널 꺼짐  ({reason})',
    '  This mode is for dumb consoles, so the phone channel is unused.\n'
    '  Your settings are intact; run without --plain/--ascii to turn it back on.':
        '  이 모드는 dumb 콘솔용이라 폰 채널을 쓰지 않습니다.\n'
        '  설정은 그대로이며, --plain/--ascii 없이 실행하면 다시 켜집니다.',
    'No remote connection': '원격 연결 없음',
    '  No Telegram bot token.\n'
    '  Fill in the slots in {path}:\n'
    '    TELEGRAM_BOT_TOKEN=...   (one line per bot; @BotFather issues them)\n'
    '    TELEGRAM_CHAT_ID=...     (your own numeric id; @userinfobot tells you)\n'
    '  (Environment variables work too. Restart the session afterwards.)':
        '  텔레그램 봇 토큰이 없습니다.\n'
        '  {path} 의 빈 슬롯을 채우세요:\n'
        '    TELEGRAM_BOT_TOKEN=...   (봇 하나당 한 줄. @BotFather 가 발급)\n'
        '    TELEGRAM_CHAT_ID=...     (본인 숫자 id. @userinfobot 이 알려줌)\n'
        '  (환경변수로 줘도 됩니다. 설정 후 세션 재시작.)',
    '  Fill in the slots in {path} and try again:':
        '  {path} 의 빈 슬롯을 채우고 다시 시도하세요:',
    'remote: off — no bot tokens yet. Wrote a starter file with 4 slots: {path}':
        'remote: 꺼짐 — 봇 토큰이 아직 없습니다. 슬롯 4개짜리 시작 파일을 만들었습니다: {path}',
    # 처음 생성되는 remote.env 의 내용. 파일 안에서 읽히는 글이라
    # 화면 문구보다 길어도 된다 — 손댈 곳이 바로 그 자리다.
    '# codemason remote channel — drive this session from Telegram.\n'
    '#\n'
    '# 1. In Telegram, talk to @BotFather and send /newbot. It replies with\n'
    '#    a token. Paste one per slot below. Blank slots are ignored.\n'
    '# 2. Every codemason you start claims one FREE bot by itself — no flag,\n'
    '#    no name to remember. Four slots means four windows at the same\n'
    '#    time, anywhere: this machine, a VM, another desktop. Want more at\n'
    '#    once? Add more lines.\n'
    '# 3. Send /start to each bot once, so it is allowed to message you.\n'
    '# 4. TELEGRAM_CHAT_ID is your own numeric id — ask @userinfobot for it.\n'
    '#\n'
    '# This file holds your bot tokens. Anyone who can read it can read your\n'
    '# alerts and type into your sessions, so keep it mode 600.\n':
        '# codemason 원격 채널 — 텔레그램에서 이 세션을 조종합니다.\n'
        '#\n'
        '# 1. 텔레그램에서 @BotFather 에게 /newbot 을 보내면 토큰을 줍니다.\n'
        '#    아래 슬롯에 한 줄에 하나씩 붙여넣으세요. 빈 줄은 무시됩니다.\n'
        '# 2. codemason 을 켤 때마다 비어 있는 봇을 알아서 하나 잡습니다.\n'
        '#    옵션도, 외울 이름도 없습니다. 슬롯 4개면 창 4개를 동시에 쓸 수\n'
        '#    있고, 어디서든 됩니다 — 이 PC, VM, 다른 데스크탑. 더 필요하면\n'
        '#    줄을 더 추가하세요.\n'
        '# 3. 각 봇에게 /start 를 한 번 보내야 봇이 나에게 말을 걸 수 있습니다.\n'
        '# 4. TELEGRAM_CHAT_ID 는 본인의 숫자 id 입니다 — @userinfobot 에게 물어보세요.\n'
        '#\n'
        '# 이 파일에는 봇 토큰이 들어갑니다. 읽을 수 있는 사람은 내 알림을 보고\n'
        '# 내 세션에 입력까지 할 수 있으니 권한은 600 으로 두세요.\n',
    'Remote connected  (Telegram)': '원격 연결됨  (텔레그램)',
    '  This session : {label}{state}': '  이 세션   : {label}{state}',
    '(unregistered)': '(미등록)',
    '  <- listening': '  ← 수신 중',
    '  (another session is listening)': '  (다른 세션이 수신 중)',
    '  Others       : {names}': '  다른 세션 : {names}',
    'What you can do from the phone': '폰에서 할 수 있는 것',
    '  Send a message   queued for this session (mid-turn ones wait their turn)':
        '  메시지 전송   보내면 이 세션 큐로 들어갑니다 (턴 중이면 대기 후 처리)',
    '  Approve tools    risky tools raise buttons; first answer wins':
        '  권한 승인     위험한 툴은 폰에 버튼이 뜹니다. 먼저 답한 쪽이 이깁니다',
    '  Answer questions choices arrive as buttons (free text stays local)':
        '  질문 답변     선택지도 폰에 버튼으로 뜹니다 (자유 입력은 터미널)',
    '  /start           lists the running sessions': '  /start        폰에서 치면 실행 중인 세션 목록',
    '  Reply            replying to an alert routes to that session':
        '  답장          세션 알림에 답장하면 그 세션으로 갑니다',
    'Settings': '설정',
    '  Notify after     only turns longer than {secs:g}s{every}   CODEMASON_NOTIFY_AFTER':
        '  알림 임계     {secs:g}초 이상 걸린 턴만 알림{every}   CODEMASON_NOTIFY_AFTER',
    ' (0 = every turn)': ' (0=매번)',
    '  Approval wait    re-asks every {mins:g} min   CODEMASON_ASK_TIMEOUT':
        '  승인 대기     {mins:g}분마다 재요청   CODEMASON_ASK_TIMEOUT',
    'Still no response — retrying ({mins} min left). Press Esc or Ctrl+C to stop.':
        '여전히 응답이 없어 재시도 중입니다 ({mins}분 남음). 중단하려면 Esc 또는 Ctrl+C.',
    'Model server not responding ({err}) — retrying for up to {mins} min. '
    'Press Esc or Ctrl+C to stop.':
        '모델 서버가 응답하지 않습니다 ({err}) — 최대 {mins}분 동안 재시도합니다. '
        '중단하려면 Esc 또는 Ctrl+C.',
    'Still waiting on an answer — asking again ({left} min left before choosing).':
        '아직 답이 없어 다시 요청합니다 (스스로 선택하기까지 {left}분 남음).',
    'No answer after {mins} min — going with the recommended option: {label}':
        '{mins}분 동안 답이 없어 추천 항목으로 진행합니다: {label}',
    '  A turn started from the phone always gets its answer.':
        '  폰에서 온 턴은 임계와 무관하게 항상 답을 보냅니다.',
}

MESSAGES.update({
    # 원격 입력 (remote/input.py)
    '{count} message(s) not delivered': '전달 못한 메시지 {count}건',
    'The session ended before these could be handled:': '세션이 종료되어 처리하지 못했습니다:',
    'Other sessions are still running: {names}': '다른 세션이 계속 실행 중입니다: {names}',
    'No sessions are running. Messages will not be delivered until you start one.':
        '실행 중인 세션이 없습니다. 새 세션을 시작하기 전까지 메시지는 전달되지 않습니다.',
    'Session ended: {name}': '세션 종료: {name}',
    '{n}m ago': '{n}분 전',
    '{n}h ago': '{n}시간 전',
    '{n}d ago': '{n}일 전',
    '… and {n} more': '… 외 {n}건',
    '{count} older message(s) — not run': '오래된 메시지 {count}건 — 실행하지 않음',
    'These arrived while no session was listening. Running them now, in a '
    'different context, would be risky, so they were held:':
        '세션이 없던 동안 도착했습니다. 지금 실행하면 맥락이 달라 위험해 보류했습니다:',
    'Send them again if you still want them.': '필요하면 다시 보내주세요.',
    'No sessions are running': '실행 중인 세션이 없습니다',
    'Your message is held and will be delivered when one starts.':
        '메시지는 보관해 두었습니다. 세션을 시작하면 전달됩니다.',
    'Answer delivered': '응답 전달됨',
    'approved': '승인',
    'denied': '거부',
    'Which session should this go to?': '어느 세션으로 보낼까요?',
    'no inbound link (this session does not use remote input)':
        '수신 연결 없음 (이 세션은 원격 입력을 쓰지 않습니다)',
    'inbound already attached': '수신 이미 연결됨',
    'inbound attached': '수신 연결됨',
    'inbound failed to attach': '수신 연결 실패',
    'inbound already detached': '수신 이미 끊김',
    'inbound detached (poller stopped, poll lock released)': '수신 끊음 (폴러 정지, 폴 락 해제)',
    'failed to detach inbound: {error}': '수신 정지 실패: {error}',
})

MESSAGES.update({
    # 폰 알림 · 권한 · 질문 (remote/)
    'Turn done': '턴 완료',
    'Context {pct}%': '컨텍스트 {pct}%',
    'images {k}/{n}': '이미지 {k}/{n}',
    'images {k}/{n} ({d} dropped)': '이미지 {k}/{n} ({d}장 빠짐)',
    'Turn failed': '턴 실패',
    'You asked': '요청',
    '✅ Allow': '✅ 승인',
    '⛔ Deny': '⛔ 거부',
    'Permission needed: {tool}': '권한 필요: {tool}',
    'Permission needed (re-asking · {mins} min so far): {tool}':
        '권한 필요 (재요청 · {mins}분째): {tool}',
    '{title} (re-asking · {mins} min so far)': '{title} (재요청 · {mins}분째)',
    'Question: {header}': '질문: {header}',
    'Question': '질문',
    'Forwarded': '전달됨',
    '✏️ Other (type your answer)': '✏️ 기타 (직접 입력)',
    'Type your answer and send it as a message': '답을 메시지로 입력해서 보내세요',
    'Several questions are waiting': '답을 기다리는 질문이 여러 개입니다',
    'Reply to the question you are answering, then send it again.': '답할 질문에 답장(Reply)으로 다시 보내 주세요.',
    'This request has expired': '만료된 요청입니다',
    'Running sessions:': '실행 중인 세션:',
    'Just send a message and it goes there.': '메시지를 그냥 보내면 전달됩니다.',
    '<b>Reply</b> to a session alert to route there.\n'
    'Send without replying and you will be asked which one.':
        '세션 알림에 <b>답장</b>하면 그 세션으로 갑니다.\n그냥 보내면 어느 세션인지 물어봅니다.',
    'No sessions are running.\nMessages are held and delivered once one starts.':
        '실행 중인 세션이 없습니다.\n보낸 메시지는 보관해 두었다가 세션이 시작되면 전달됩니다.',
    'Messages sent before a session started are not delivered':
        '세션 시작 전 메시지는 전달하지 않습니다',
    'What you sent while nothing was listening will not be run. '
    'Send it again if you still want it.':
        '세션이 없던 동안 보낸 메시지는 실행하지 않습니다. 필요하면 다시 보내주세요.',
    'working…': '작업 중…',
    '… the full {chars:,} characters were sent as `{name}`.':
        '… 전문 {chars:,}자는 첨부 파일 `{name}` 로 보냈습니다.',
    'Routed': '전달됨',
    'answered at the terminal': '터미널에서 응답됨',
    'waiting… (a fresh prompt is on its way)': '대기 중… (새 요청을 보냅니다)',
})

MESSAGES.update({
    # 터미널 (cli.py)
    '(remote)': '(원격)',
    '(remote · {n}m ago)': '(원격 · {n}분 전)',
    '(remote · {n}h ago)': '(원격 · {n}시간 전)',
    '(remote · {n}d ago)': '(원격 · {n}일 전)',
    'Cannot turn the remote channel on — no credentials.':
        '원격 채널을 켤 수 없습니다 — 자격증명이 없습니다.',
    'Remote channel on ({saved}) — but still off right now because of {reason}.':
        '원격 채널 on ({saved}) — 다만 지금은 {reason} 때문에 여전히 꺼져 있습니다.',
    'Remote channel on ({saved}). Alerts, approvals and questions go to the phone.':
        '원격 채널 on ({saved}). 폰으로 알림·승인·질문이 갑니다.',
    'Remote channel off ({saved}). Alerts, spinner, approvals, questions and '
    'inbound messages all stop.':
        '원격 채널 off ({saved}). 알림·스피너·승인·질문·수신이 모두 멈춥니다.',
    'Usage: /remote [on | off]   (no argument shows status)':
        'Usage: /remote [on | off]   (인자 없으면 상태 표시)',
    'Skip approvals {state} ({saved}). ': '승인 생략 {state} ({saved}). ',
    'Tools run without asking.': '도구 호출을 묻지 않고 진행합니다.',
    'Every tool call asks for approval.': '도구 호출마다 승인을 요청합니다.',
    'Plan approval (ExitPlanMode) and questions (AskUserQuestion) still ask either way.':
        '계획 승인(ExitPlanMode)과 선택 요구(AskUserQuestion)는 어느 쪽이든 물어봅니다.',
    'Skip approvals is {state}.': '승인 생략은 {state} 입니다.',
    '  Tools are auto-approved.': '  도구 호출 자동 허가.',
    '  Every tool call asks.': '  도구 호출마다 승인 요청.',
})

MESSAGES.update({
    # 질문 UI (arrow_picker / AskUserQuestion)
    '↑/↓ move · enter select · esc cancel': '↑/↓ 이동 · enter 선택 · esc 취소',
    '↑/↓ move · space toggle · enter confirm · esc cancel':
        '↑/↓ 이동 · space 토글 · enter 확정 · esc 취소',
    'Other (type your own answer)': '기타 (직접 입력)',
    'Your answer: ': '직접 입력: ',
    'Select {hint}: ': '{hint} 선택: ',
    'numbers comma-separated': '번호를 쉼표로 구분해',
    'a number': '번호를',
    # 권한 프롬프트
    'Allow? (y/n, a=always): ': '허용할까요? (y/n, a=항상): ',
    "Please answer 'y', 'n' or 'a'.": "'y', 'n', 'a' 중에 답해주세요.",
})

MESSAGES.update({
    'Question': '질문',
})

MESSAGES.update({
    # SessionSearch 결과 요약 줄
    'No matches in past sessions': '지난 세션에서 일치 항목 없음',
    '{hits} match(es) in {sessions} past session(s)':
        '지난 세션 {sessions}개에서 {hits}건 일치',
    # config.json 읽기·검증·이전 (utils/config.py)
    'saved': '저장됨',
    'not saved: config.json could not be read': '저장 안 됨: config.json 을 읽지 못함',
    '[codemason] Could not read {path}: {error}':
        '[codemason] {path} 를 읽지 못했습니다: {error}',
    '[codemason] Running on defaults. Settings changed in this session will not be saved '
    'until the file is fixed (it is left untouched).':
        '[codemason] 기본값으로 실행합니다. 파일을 고칠 때까지 이 세션에서 바꾼 설정은 저장되지 않습니다'
        '(파일은 그대로 둡니다).',
    '[codemason] config.json: "{key}" = {value} is not a valid value for this setting; '
    'using the default ({default}).':
        '[codemason] config.json: "{key}" = {value} 는 이 설정에 쓸 수 없는 값이라 기본값({default})을 씁니다.',
    '[codemason] config.json updated: "{key}" {old} -> {new} (an old default that has since been fixed).':
        '[codemason] config.json 갱신: "{key}" {old} -> {new} (나중에 고쳐진 옛 기본값).',
    '[codemason] Not saved: {path} could not be read when this session started. '
    'Fix the file and restart; your change applies to this session only.':
        '[codemason] 저장하지 않았습니다: 이 세션을 시작할 때 {path} 를 읽지 못했습니다. '
        '파일을 고치고 다시 시작하세요. 이번 변경은 이 세션에만 적용됩니다.',
})

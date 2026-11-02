# codemason (Python)

> ## ⚠️ 고지 · Disclaimer
>
> 이 저장소는 **있는 그대로(as-is)** 공개됩니다. 어떤 보증도 하지 않습니다.
>
> - 이 코드를 사용해 발생하는 **모든 결과와 책임은 사용자 본인**에게 있습니다.
> - 이 도구는 파일을 읽고 **고치고 지우며, 셸 명령을 실행합니다.** 무엇을 실행할지는
>   모델이 정합니다. 중요한 작업 트리에서는 쓰지 마시고, 버전 관리 아래에서만 쓰세요.
> - 관할 **법령과 제3자 약관(모델·API·서비스 라이선스 포함)을 지킬 책임은 사용자**에게 있습니다.
>
> *Provided **as-is**, without warranty of any kind. This tool edits files and executes
> shell commands under model control. See [LICENSE](LICENSE).*

---

터미널에서 도는 코딩 에이전트 CLI. **로컬 LLM**(vLLM / llama.cpp)을 상대로 돌며
클라우드에 의존하지 않는다. Rust 포팅판은 [AmesianX/codemason](https://github.com/AmesianX/codemason).

```
148개 .py 파일 / 52,299줄 · 테스트 63개 · Python 3.10+
```

## 설치

```bash
./setup.sh                 # 리눅스·macOS
setup.bat                  # 윈도우 (유저 PATH 에 등록, 새 터미널부터 적용)
```

의존성: `httpx`, `prompt-toolkit`, `pydantic`, `tokenizers`, `markdownify`,
`ddgs`, `rich`, `pyyaml`, `cryptography`, `mcp`, `pillow`.

`codemason/vendor/ripgrep/` 에 플랫폼별 ripgrep 이 동봉돼 있다.
Glob/Grep 은 `시스템 rg → 동봉 바이너리 → 네이티브 워크` 순으로 해석한다.

## 사용

```bash
codemason                  # REPL — 타이핑은 큐잉되고 작업 중에도 입력을 받는다
codemason -p "..."         # 프린트 모드. 한 번 묻고 결과만 출력
codemason --help
```

## 구성

```
codemason/
├── cli.py             REPL UI
├── query.py           에이전트 루프 — 모델 호출, 도구 실행, 자동 압축
├── print_runner.py    -p 모드 실행기
├── task.py / tasks/   백그라운드 작업
├── tools/             Read · Write · Edit · Bash · Glob · Grep · Agent ·
│                      WebFetch · MCP · TodoWrite …
├── services/          백엔드 어댑터 (vllm, llamacpp)
├── remote/            원격 봇 풀
├── state/             세션 · 재개 · 트랜스크립트
├── commands/          슬래시 명령
├── constants/         프롬프트와 스키마
├── utils/             설정, 파일, 포맷, 완성 …
└── locales/           다국어 (영어 원문이 키)
```

### 동작 개요

1. **입력** — REPL 이 입력을 큐에 넣는다. 모델이 도는 중에도 타이핑이 막히지 않는다.
2. **에이전트 루프** (`query.py`) — 모델을 부르고, 도구 호출을 실행하고, 결과를 다시
   넣는다. 대화가 길어지면 자동 압축하고, 같은 호출이 반복되면 제동을 건다.
3. **도구** (`tools/`) — 각 도구가 스키마와 실행부를 함께 가진다. 파일 수정은 읽기를
   선행 조건으로 두고, 셸은 권한 확인을 거친다.
4. **상태** (`state/`) — 세션을 트랜스크립트로 남겨 재개할 수 있게 한다.

## 라이선스

MIT. [LICENSE](LICENSE) 참조.

Claude Code 의 동작을 참고해 독자 구현했으며, 원 제품과 제휴 관계가 없다.

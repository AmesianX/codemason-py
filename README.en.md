# codemason (Python)

[한국어](README.md) | **English**

**codemason is maintained as one implementation in two languages.** This repository is the
**Python edition**; the Rust edition lives at [AmesianX/codemason](https://github.com/AmesianX/codemason).
A fix on one side is carried to the other — features and behavior are kept in step.

> ## ⚠️ Disclaimer
>
> This repository is published **as-is**, without warranty of any kind.
>
> - **All consequences of and responsibility for** using this code rest with **you**.
> - This tool **reads, edits and deletes files and runs shell commands.** The model decides
>   what to run. Do not use it on an important working tree — only under version control.
> - **You are responsible for complying with applicable law and third-party terms**
>   (including model, API and service licenses).
>
> See [LICENSE](LICENSE).

A coding-agent CLI that runs in the terminal against a **local LLM** (vLLM / llama.cpp),
with no cloud dependency.

```
148 .py files / 52,365 lines · 64 tests · Python 3.10+
```

## Highlights

The matching server is [dspark-serve](https://github.com/AmesianX/dspark-serve)
(GLM-5.3-Flash on two DGX Sparks, TP2). Items marked ⓢ are server features the client uses.

| Feature | What it does |
|---|---|
| **Effectively unlimited prefix cache** ⓢ | The server keeps prefix KV evicted from the GPU on 200 GB of NVMe per node, and it survives a restart. Continuing a long session, or reopening it with `/resume`, picks up from the cache instead of re-reading the prompt. |
| **Per-request `/ab`** ⓢ | `/ab off\|on\|auto` — runtime refusal removal (a rank-24 adapter on o_proj). It rides on each request as `vllm_xargs.ab_scale`; the server has no global switch (`POST /ab` returns 410). Your setting has zero effect on anyone else's output on the same server — verified by bit-identical logprobs of co-batched neighbor requests; KV computed with it on lives in a per-scale cache namespace that original-model requests never read (while it still reuses the shared original prefix, so `/ab on` stays cached). `auto` turns it on only for the first request of a user turn (the one that decides whether to accept), because abliteration removes not only refusal but also the ability to say "that doesn't exist", and fabrication rises (4/60 → 18–22/60). |
| **Per-session cache isolation** ⓢ | Each install sends a random session id (`~/.codemason/cache_session`) inside the encrypted body; the gateway turns it into the cache salt with a server-only secret — nobody can reuse or time another user's cache. The system prompt goes out as a public message (instructions, tool usage) and a private one (memory, environment, operator directive); only the public part is one shared cache entry for everyone. |
| **Cold prefill only** | The spinner shows `⟲ N re-read` only when the server recomputes input that was already sent (a broken cache, a restart, a resumed session). New input — search results, the next question — is not shown. `-dbg` shows new input, re-read and cache hits. |
| **Parallel sub-agents + priority** ⓢ | Up to 3 at once. Sub-agent requests carry vLLM priority 1, so the main conversation is served first from the queue. Sub-agent context budget: 30% of the parent's threshold each, 90% combined. |
| **Deterministic output** ⓢ | At temperature 0 the output is the same regardless of batch composition, load or restarts (server batch invariance). |
| **Web search** | Self-hosted SearXNG first; falls back to duckduckgo, brave, mojeek and google directly. When results are thin or engines are blocked, the server escalates to an identity-bearing browser search on its own — the client cannot request it. |
| **Image input** ⓢ | Up to 5 images per request. Photos received over Telegram are passed through as-is. |
| **Encrypted gateway** ⓢ | For remote access the request body is encrypted with AES-GCM. The key is never sent — holding the key is the credential. |
| **Remote (Telegram)** | `/remote` — give instructions, approve tools and answer questions from a phone. Long results arrive as files. |
| **Unattended runs** | `/goal` (a goal injected every turn) + completion gates (`/goal gate add <cmd>`) + `/keepgoing`. |
| **Exact token counts** | Ships the served model's tokenizer, so auto-compaction (at 88% of the context), the window check and usage are real counts, not estimates. |
| **Per-model sampling** | `sampling_profile: "auto"` — GLM gets the vendor's recommended values (temperature 1.0, top_p 0.95, no top_k) and config.json's values are ignored. `"manual"` sends config.json as-is. |
| **Multilingual UI** | `/language`. English source strings are the translation keys. |

## Install

```bash
./setup.sh                 # Linux/macOS
setup.bat                  # Windows (adds to the user PATH; applies to new terminals)
```

Dependencies: `httpx`, `prompt-toolkit`, `pydantic`, `tokenizers`, `markdownify`,
`ddgs`, `rich`, `pyyaml`, `cryptography`, `mcp`, `pillow`.

A per-platform ripgrep is bundled in `codemason/vendor/ripgrep/`.
Glob/Grep resolve it in the order `system rg → bundled binary → native walk`.

## Usage

```bash
codemason                  # REPL — typing is queued and never blocks while the model runs
codemason -p "..."         # print mode: ask once, print only the result
codemason --help
```

## Inference engine recommendation

> **Running on llama.cpp is implemented, but for performance and completeness we strongly
> recommend against it. This project recommends vLLM.**

This project's b12x kernels exist only for vLLM; in the project's environment they give a
comfortable prefill of 1,300+ tokens/s and decoding of 35–40+ tokens/s. llama.cpp, even after
heavy optimization, stayed around 450+ tokens/s prefill and 25–30 tokens/s decoding — a
clearly noticeable difference.

The llama.cpp setup we used also had no built-in two-node tensor parallelism; we implemented
it ourselves and optimized further, and still did not reach acceptable responsiveness.

So development and optimization now focus on vLLM. llama.cpp has only basic execution
implemented, without further features or sufficient testing, and is not recommended.

## Layout

```
codemason/
├── cli.py             REPL UI
├── query.py           agent loop — model calls, tool execution, auto-compaction
├── print_runner.py    -p mode runner
├── task.py / tasks/   background tasks
├── tools/             Read · Write · Edit · Bash · Glob · Grep · Agent ·
│                      WebFetch · MCP · TodoWrite …
├── services/          backend adapters (vllm, llamacpp)
├── remote/            remote bot pool
├── state/             sessions · resume · transcripts
├── commands/          slash commands
├── constants/         prompts and schemas
├── utils/             config, files, formatting, completion …
└── locales/           translations (English source string is the key)
```

### How it works

1. **Input** — the REPL queues input. Typing never blocks while the model runs.
2. **Agent loop** (`query.py`) — calls the model, executes tool calls, and feeds the results
   back. Long conversations are compacted automatically, and repeated identical calls are braked.
3. **Tools** (`tools/`) — each tool carries its schema and its executor. File edits require a
   prior read; the shell goes through permission checks.
4. **State** (`state/`) — sessions are kept as transcripts so they can be resumed.

## License

MIT. See [LICENSE](LICENSE).

Independently implemented with reference to the behavior of Claude Code; not affiliated
with the original product.

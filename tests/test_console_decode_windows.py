"""Native Windows must not mangle the UTF-8 half of a console stream.

A Windows program and cmd.exe itself write into ONE stream: the first often in
UTF-8, the second always in the console codepage. decode_console_output's
is_windows() branch used to decode the whole buffer with the OEM page and
return, so the run-by-run decoder below it was reached only OFF Windows (under
WSL, where the OEM page cannot be asked for). The result was the WSL bug
mirrored: there the legacy half became U+FFFD, here the legacy half came
through and pktmon's UTF-8 Korean arrived as mojibake.

Windows-only on purpose. The POSIX path has its own coverage through
_decode_mixed_console_output and the configured legacy encoding, and the OEM
codepage this file is about does not exist there -- so these cases would be
asserting something the platform cannot do. They skip rather than pretend.

Run directly (exit 0 = all pass).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

if os.name != 'nt':
    print('SKIP  console OEM decoding is Windows-only (no "oem" codec here)')
    raise SystemExit(0)

from codemason.utils.windows_shell import decode_console_output  # noqa: E402

results = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name
          + (('\n      ' + detail) if detail and not ok else ''))
    results.append(bool(ok))


# ── one stream, both encodings ──────────────────────────────────────────────
#
# Exactly the fixture the Rust build pins in
# utils::windows_shell::mixed_console_decode_tests, so the two builds are
# known to agree about the same bytes.
UTF8_HEAD = 'PktMon 드라이버와 통신하지 못했습니다.\n'
LEGACY = 'UNC 경로는 지원되지 않습니다.'
UTF8_TAIL = '\netl2pcap 로그 파일을 변환합니다.\n'

buf = UTF8_HEAD.encode('utf-8') + LEGACY.encode('cp949') + UTF8_TAIL.encode('utf-8')
assert buf.decode('utf-8', errors='ignore') != buf.decode('utf-8', errors='replace'), \
    'fixture must not be valid UTF-8, or it takes the fast path and says nothing'

out = decode_console_output(buf)
check('the UTF-8 runs survive a stream that also carries CP949',
      UTF8_HEAD.strip() in out and UTF8_TAIL.strip() in out, repr(out[:200]))
check('the CP949 run survives too', LEGACY in out, repr(out[:200]))
check('and nothing in it was replaced', '�' not in out, repr(out[:200]))

# ── the cases the fix must not break ───────────────────────────────────────
pure = 'ls: 그런 파일이나 디렉터리가 없습니다\n'
check('pure UTF-8 is returned byte-for-byte',
      decode_console_output(pure.encode('utf-8')) == pure)

# "정보:" as tasklist writes it on a Korean install: no UTF-8 at all.
check('a line that is entirely CP949 still decodes',
      decode_console_output('정보:'.encode('cp949')) == '정보:',
      repr(decode_console_output('정보:'.encode('cp949'))))

# Binary is not text in disguise: bytes that decode to control characters must
# not be reinterpreted as hangul. `cat` of a JPEG still looks like a JPEG.
jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46])
check('binary keeps its lossy form rather than being guessed into text',
      '�' in decode_console_output(jpeg),
      repr(decode_console_output(jpeg)))

print()
print(f'{sum(results)}/{len(results)} pass')
sys.exit(0 if all(results) else 1)

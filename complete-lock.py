#!/usr/bin/env python3
"""Complete uv.lock for pure Nix evaluation.

Flat / find-links indexes (e.g. PyG's data.pyg.org) serve no hashes, so `uv lock`
leaves those wheels without a sha256 and pure Nix eval refuses the fetch
(astral-sh/uv#10987). uv preserves manually-added hashes, so we fetch the wheels
this x86_64-linux venv would use (linux_x86_64 + none-any) and inject their
sha256 into uv.lock. Tool-agnostic: the value is the wheel file's sha256 — the
same one pip / uv / sha256sum produce — so no Nix is required.

uv.lock is parsed with tomllib to *identify* hashless wheels structurally, and the
hash is inserted with a literal string replace so uv's own formatting is kept
intact (a full TOML rewrite would reformat the whole lock).

Idempotent. Run after `uv lock`:
    uv lock && ./complete-lock.py
    ./complete-lock.py --check   # CI gate: nonzero exit if any hash is missing

Needs Python 3.11+ (tomllib).
"""

import hashlib
import subprocess
import sys
import tomllib
import urllib.request

# Wheel filename suffixes this x86_64-linux build can actually fetch: the platform
# wheels and universal (none-any) wheels. Other-platform siblings in the lock
# (win_amd64, macosx, …) are never fetched here, so they need no hash.
WANTED = ("linux_x86_64.whl", "none-any.whl")


def lock_path() -> str:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{root}/uv.lock"


def hashless_urls(lock_text: str) -> list[str]:
    """URLs of fetchable wheels that have a `url` but no `hash` (structural)."""
    data = tomllib.loads(lock_text)
    return [
        wheel["url"]
        for pkg in data.get("package", [])
        for wheel in pkg.get("wheels", [])
        if "url" in wheel and "hash" not in wheel and wheel["url"].endswith(WANTED)
    ]


def sha256_of(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return hashlib.sha256(resp.read()).hexdigest()


def main() -> int:
    check = len(sys.argv) > 1 and sys.argv[1] == "--check"
    path = lock_path()
    with open(path, encoding="utf-8") as f:
        text = f.read()

    urls = hashless_urls(text)
    if not urls:
        print("complete-lock: all flat-index wheels already hashed", file=sys.stderr)
        return 0
    if check:
        print(f"complete-lock: {len(urls)} wheel(s) missing a hash in {path}", file=sys.stderr)
        return 1

    for url in urls:
        digest = sha256_of(url)
        old = f'{{ url = "{url}" }}'
        new = f'{{ url = "{url}", hash = "sha256:{digest}" }}'
        if old not in text:
            print(f"complete-lock: ERROR — no lock entry {old!r}", file=sys.stderr)
            return 1
        text = text.replace(old, new, 1)
        print(f"  + {url.rsplit('/', 1)[-1]}  sha256:{digest[:12]}…", file=sys.stderr)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    # Structural re-check: nothing this build fetches should remain hashless.
    if hashless_urls(text):
        print("complete-lock: ERROR — wheels still hashless after injection", file=sys.stderr)
        return 1
    print(f"complete-lock: injected {len(urls)} flat-index wheel hash(es) into {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stable semantic code anchors for numerical findings."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.]*|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|"
    r"==|!=|<=|>=|:=|\*\*|//|[-+*/%@<>=()[\]{},.:]",
    re.IGNORECASE,
)


def _repo_path(workspace_root: Path, repo: str) -> Path:
    """Find a source checkout in the active flat or family workspace."""
    root = Path(workspace_root)
    resolver = root / "PyAutoBrain" / "agents" / "_repo_paths.py"
    if resolver.is_file():
        spec = importlib.util.spec_from_file_location("_pyauto_repo_paths", resolver)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.repo_path(root, repo, required=True)
    flat = root / repo
    if (flat / ".git").exists():
        return flat
    if any(
        (family / repo).exists()
        for family in root.iterdir()
        if family.is_dir() and not (family / ".git").exists()
    ):
        raise RuntimeError(f"Grouped checkout {repo} needs PyAutoBrain repo resolver")
    return flat


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Tokens stable to whitespace and comments, for code and config snippets."""

    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return tuple(token.lower() for token in _TOKEN_RE.findall(uncommented))


def token_fingerprint(tokens: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


@dataclass(frozen=True)
class CodeAnchor:
    repo: str
    commit: str
    path: str
    start_line: int
    end_line: int
    token_fingerprint: str
    tokens: tuple[str, ...]
    symbol: str | None = None
    config_key: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tokens"] = list(self.tokens)
        return data

    def status_in_source(self, source: str) -> str:
        """Return unchanged, moved, changed, or missing for current source text."""

        lines = source.splitlines()
        current = "\n".join(lines[self.start_line - 1 : self.end_line])
        if token_fingerprint(normalized_tokens(current)) == self.token_fingerprint:
            return "unchanged"
        all_tokens = normalized_tokens(source)
        wanted = self.tokens
        if wanted and any(
            all_tokens[i : i + len(wanted)] == wanted
            for i in range(0, len(all_tokens) - len(wanted) + 1)
        ):
            return "moved"
        if 0 < self.start_line <= len(lines):
            return "changed"
        return "missing"


def repo_commit(workspace_root: Path, repo: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_repo_path(workspace_root, repo)), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def anchor_from_pattern(
    workspace_root: Path,
    *,
    repo: str,
    path: str,
    pattern: str,
    before: int = 0,
    after: int = 0,
    symbol: str | None = None,
    config_key: str | None = None,
) -> CodeAnchor:
    """Capture a small expression/config anchor located by a unique pattern."""

    source_path = _repo_path(workspace_root, repo) / path
    lines = source_path.read_text().splitlines()
    matches = [index for index, line in enumerate(lines) if pattern in line]
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r} in {repo}/{path}, found {len(matches)}")
    index = matches[0]
    start = max(0, index - before)
    end = min(len(lines), index + after + 1)
    tokens = normalized_tokens("\n".join(lines[start:end]))
    return CodeAnchor(
        repo=repo,
        commit=repo_commit(workspace_root, repo),
        path=path,
        start_line=start + 1,
        end_line=end,
        token_fingerprint=token_fingerprint(tokens),
        tokens=tokens,
        symbol=symbol,
        config_key=config_key,
    )


def maybe_anchor_from_pattern(
    workspace_root: Path,
    **kwargs,
) -> CodeAnchor | None:
    """Capture an anchor when present; absence must not define finding persistence."""

    try:
        return anchor_from_pattern(workspace_root, **kwargs)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None

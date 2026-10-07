"""Standard-library source routing; no scientific imports or execution on lookup.

Cell IDs and output names retain their historical spelling. Source paths are a
separate identity. The versioned JSON is also readable by Brain without importing
this project or scraping executable sweep source.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def load_routes(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "catalogue/script_routes.json").read_text())
    if manifest.get("schema") != "profiling-script-routes" or manifest.get("version") != 1:
        raise ValueError("Unsupported profiling script routes")
    seen = set()
    targets = set()
    for row in manifest["routes"]:
        for key in ("legacy", "path"):
            path = PurePosixPath(row[key])
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "scripts":
                raise ValueError(f"Unsafe script route: {row[key]}")
            if path.suffix != ".py" or (key == "path" and not (root / path).is_file()):
                raise ValueError(f"Missing script route: {row[key]}")
        if row["legacy"] in seen or row["path"] in targets:
            raise ValueError("Duplicate script route")
        seen.add(row["legacy"])
        targets.add(row["path"])
    return manifest


def canonical_path(path: str | Path, root: Path = ROOT) -> Path:
    """Resolve an old or canonical repository-relative path, failing on unknowns."""
    path = Path(path)
    relative = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
    for row in load_routes(root)["routes"]:
        if relative in (row["legacy"], row["path"]):
            return root / row["path"]
    raise ValueError(f"Unknown profiling script route: {relative}")


def legacy_stem(path: str | Path, root: Path = ROOT) -> str:
    """Historical filename for output markers; never derive it from the new name."""
    canonical = canonical_path(path, root)
    return next(
        Path(row["legacy"]).stem
        for row in load_routes(root)["routes"]
        if root / row["path"] == canonical
    )


def runtime_path(dataset: str, model: str, root: Path = ROOT) -> Path:
    """Runtime route for legacy cell IDs or the canonical rectangular alias."""
    if model == "rectangular":
        model = "pixelization"
    prefix = (
        "interferometer/likelihood_runtime/datacube"
        if dataset == "datacube"
        else f"{dataset}/likelihood_runtime"
    )
    return canonical_path(f"scripts/{prefix}/{model}.py", root)

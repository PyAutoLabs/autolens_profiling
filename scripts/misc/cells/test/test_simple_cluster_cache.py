"""Cache gate regression for the legacy simple cluster in profiling issue 445."""

import sys
from pathlib import Path

import autolens as al
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from _profile_cli import auto_simulate_if_missing  # noqa: E402


def test_simple_cluster_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset = tmp_path / "dataset" / "cluster" / "simple"
    dataset.mkdir(parents=True)

    def check_cache():
        auto_simulate_if_missing(
            dataset, dataset_type="cluster", instrument="simple", workspace_root=tmp_path
        )

    with pytest.raises(FileNotFoundError, match="scaling_galaxies.csv"):
        check_cache()

    (dataset / "scaling_galaxies.csv").touch()
    truth = dataset / "tracer.json"
    truth.write_text("{invalid")
    with pytest.raises(ValueError, match="tracer.json"):
        check_cache()

    al.output_to_json(obj=al.Tracer(galaxies=[al.Galaxy(redshift=0.5)]), file_path=truth)
    check_cache()

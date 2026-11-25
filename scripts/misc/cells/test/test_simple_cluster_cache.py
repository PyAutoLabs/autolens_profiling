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

    (dataset / "scaling_galaxies.csv").write_text("y,x,luminosity\n0.5,-1,0.4\n")
    truth = dataset / "tracer.json"
    truth.write_text("{invalid")
    with pytest.raises(ValueError, match="tracer.json"):
        check_cache()

    al.output_to_json(obj=al.Tracer(galaxies=[al.Galaxy(redshift=0.5)]), file_path=truth)
    check_cache()


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "y,x,luminosity\n",
        "y,x\n1,2\n",
        "y,x,luminosity\n1,2,bad\n",
        "y,x,luminosity\nnan,2,1\n",
        "y,x,luminosity\n1,2,inf\n",
        "y,x,luminosity\n1,2,0\n",
    ],
)
def test_unusable_members_are_preserved(tmp_path, monkeypatch, contents):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    members = tmp_path / "scaling_galaxies.csv"
    members.write_text(contents)
    monkeypatch.setattr(
        al.util.dataset,
        "should_simulate",
        lambda path: pytest.fail("A bad cache must be preserved before the regime check"),
    )
    with pytest.raises(ValueError) as exc:
        auto_simulate_if_missing(
            tmp_path, dataset_type="cluster", instrument="simple", workspace_root=tmp_path
        )
    assert str(members) in str(exc.value)
    assert "Move the old dataset directory aside" in str(exc.value)
    assert members.read_text() == contents


def test_newly_simulated_simple_cache_is_validated(tmp_path, monkeypatch):
    dataset = tmp_path / "simple"

    def simulate(path, *args):
        path.mkdir()
        (path / "scaling_galaxies.csv").write_text("y,x,luminosity\n")

    monkeypatch.setattr("_profile_cli._run_simulator", simulate)
    with pytest.raises(ValueError, match="no member rows"):
        auto_simulate_if_missing(
            dataset, dataset_type="cluster", instrument="simple", workspace_root=tmp_path
        )

"""Unit tests for the XLA trace attribution helper (``xla_attribution.py``).

**No imaging, no GPU, and (except for two explicitly marked tests) no JAX.**
The helper is the instrument the phase-1 GPU decomposition is read through, so
it is pinned on fixtures where the right answer is known by construction rather
than measured: a hand-written HLO text whose fusion constituents are chosen to
span two stages, a hand-encoded ``StackFrameIndexProto`` whose frames are known
because this file wrote them, and a synthetic event list whose milliseconds add
up by arithmetic.

The three quiet failure modes these tests exist for:

1. **A stage map that is not total.** Every event must land in exactly one row.
   A rule that matches nothing, or two rules that both claim a frame, turns a
   measured table into a plausible one. ``test_every_event_lands_in_exactly_one_row``
   asserts the partition, not the labels.
2. **A fusion attributed from its representative metadata.** A fused kernel
   carries one ``op_name`` and a body full of others. Taking the representative
   silently assigns a mixed kernel to whichever stage XLA happened to name.
3. **Reconciliation that closes by construction.** ``sum_ms`` must be the kernel
   rows plus the measured idle and nothing else, with the wall residual reported
   as ``host_outside_span_ms`` — not folded in to make the percentage look good.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_xla_attribution.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import xla_attribution as xa  # noqa: E402

_ARRAY = "PyAutoArray/autoarray/"
_ABSTRACT = _ARRAY + "inversion/inversion/abstract.py"
_CONVOLVER = _ARRAY + "operators/convolver.py"
_DELAUNAY = _ARRAY + "inversion/mesh/interpolator/delaunay.py"
_IMAGING_ABSTRACT = _ARRAY + "inversion/inversion/imaging/abstract.py"


# ---------------------------------------------------------------------------
# A protobuf encoder, so the wire reader is tested against bytes this file made
# ---------------------------------------------------------------------------


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _len_field(field_number: int, payload: bytes) -> bytes:
    return _tag(field_number, 2) + _varint(len(payload)) + payload


def _varint_field(field_number: int, value: int) -> bytes:
    return _tag(field_number, 0) + _varint(value)


def _encode_stack_frame_index(
    file_names: list[str],
    function_names: list[str],
    file_locations: list[tuple[int, int, int, int]],
    stack_frames: list[tuple[int, int]],
) -> bytes:
    out = b""
    for name in file_names:
        out += _len_field(1, name.encode())
    for name in function_names:
        out += _len_field(2, name.encode())
    for file_id, func_id, line, column in file_locations:
        body = (
            _varint_field(1, file_id)
            + _varint_field(2, func_id)
            + _varint_field(3, line)
            + _varint_field(4, column)
        )
        out += _len_field(3, body)
    for loc_id, parent in stack_frames:
        body = _varint_field(1, loc_id) + (_varint_field(2, parent) if parent else b"")
        out += _len_field(4, body)
    return out


#: Frames the fixture HLO below refers to by ``stack_frame_id``.
#:
#: Ids are 1-based. The two log-determinant stacks (7 and 8) are the point of
#: the fixture: identical innermost frames (``_log_det_symmetric_from``, line
#: 872) and different callers, which is the only thing that tells them apart.
_FILES = [_ABSTRACT, _CONVOLVER, _DELAUNAY, _IMAGING_ABSTRACT]
_FUNCTIONS = [
    "curvature_reg_matrix",  # 1
    "_log_det_symmetric_from",  # 2
    "log_det_curvature_reg_matrix_term",  # 3
    "log_det_regularization_matrix_term",  # 4
    "_convolved_over_sampled_jax_from",  # 5
    "operated_mapping_matrix_list",  # 6
    "_walk_from_seed",  # 7
    "reconstruction",  # 8
    "_convolved_image_over_sampled_jax_from",  # 9
    "pixel_weights_delaunay_from",  # 10
]
_LOCATIONS = [
    (1, 1, 371, 15),  # 1  abstract.py:371   curvature_reg_matrix (F + lambda*H)
    (1, 2, 872, 20),  # 2  abstract.py:872   _log_det_symmetric_from (the cholesky)
    (1, 3, 894, 15),  # 3  abstract.py:894   log_det_curvature_reg_matrix_term
    (1, 4, 941, 15),  # 4  abstract.py:941   log_det_regularization_matrix_term
    (2, 5, 586, 21),  # 5  convolver.py:586  the shared rfft2
    (4, 6, 136, 20),  # 6  imaging/abstract.py:136 operated_mapping_matrix_list
    (3, 7, 295, 12),  # 7  delaunay.py:295   the visibility walk while_loop
    (1, 8, 613, 40),  # 8  abstract.py:613   the edge-subset gather
    (2, 9, 660, 15),  # 9  convolver.py:660  _convolved_image_over_sampled_jax_from
    (3, 10, 659, 18),  # 10 delaunay.py:659  pixel_weights_delaunay_from
]
_STACK_FRAMES = [
    (1, 0),  # 1  F + lambda*H
    (2, 0),  # 2  bare cholesky, no caller  -> ambiguous on purpose
    (3, 0),  # 3  log_det_curvature_reg_matrix_term
    (2, 3),  # 4  cholesky called by log_det_curvature_reg_matrix_term
    (4, 0),  # 5  log_det_regularization_matrix_term
    (2, 5),  # 6  cholesky called by log_det_regularization_matrix_term
    (6, 0),  # 7  operated_mapping_matrix_list
    (5, 7),  # 8  the rfft2 under the MAPPING-MATRIX convolution
    (9, 0),  # 9  _convolved_image_over_sampled_jax_from
    (5, 9),  # 10 the rfft2 under the IMAGE convolution
    (7, 0),  # 11 the Delaunay walk
    (8, 0),  # 12 the edge-subset gather
    (10, 1),  # 13 written in delaunay.py, CALLED from abstract.py:371
]

STACK_INDEX_BLOB = _encode_stack_frame_index(_FILES, _FUNCTIONS, _LOCATIONS, _STACK_FRAMES)
#: The same index wrapped as ``HloModuleProto.stack_frame_index`` (field 17).
MODULE_PROTO_BLOB = (
    _len_field(1, b"jit_likelihood") + _varint_field(5, 4) + _len_field(17, STACK_INDEX_BLOB)
)


#: A small but realistic optimized-HLO fixture.
#:
#: It contains: a fusion whose body spans two stages (the mixed row), a fusion
#: whose body is one stage, two Cholesky instructions that differ only in their
#: caller, two FFTs that differ only in their caller, an (n,n) add of
#: F + lambda*H, an edge-subset gather, and a while loop for the mesh walk.
FIXTURE_HLO = """\
HloModule jit_likelihood, entry_computation_layout={(f64[1500]{0})->f64[]}

%fused_computation.mixed (param_0.1: f64[1500,1500], param_1.1: f64[1500,1500]) -> f64[1500,1500] {
  %param_0.1 = f64[1500,1500]{1,0} parameter(0)
  %param_1.1 = f64[1500,1500]{1,0} parameter(1)
  %add.7 = f64[1500,1500]{1,0} add(%param_0.1, %param_1.1), metadata={op_name="jit(fn)/add" stack_frame_id=1}
  ROOT %gather.9 = f64[1400,1400]{1,0} gather(%add.7), metadata={op_name="jit(fn)/gather" stack_frame_id=12}
}

%fused_computation.walk (param_0.2: f64[1500,2]) -> f64[1500,3] {
  %param_0.2 = f64[1500,2]{1,0} parameter(0)
  %cross.1 = f64[1500,3]{1,0} multiply(%param_0.2, %param_0.2), metadata={op_name="jit(fn)/mul" stack_frame_id=11}
  ROOT %select.2 = f64[1500,3]{1,0} select(%cross.1, %cross.1, %cross.1), metadata={op_name="jit(fn)/select_n" stack_frame_id=11}
}

ENTRY %main.9 (p0.1: f64[1500]) -> f64[] {
  %p0.1 = f64[1500]{0} parameter(0), metadata={op_name="p0"}
  %fusion.mixed = f64[1400,1400]{1,0} fusion(%p0.1, %p0.1), kind=kLoop, calls=%fused_computation.mixed, metadata={op_name="jit(fn)/gather" stack_frame_id=12}
  %fusion.walk = f64[1500,3]{1,0} fusion(%p0.1), kind=kLoop, calls=%fused_computation.walk, metadata={op_name="jit(fn)/select_n" stack_frame_id=11}
  %cholesky.0 = f64[1400,1400]{1,0} cholesky(%fusion.mixed), lower=true, metadata={op_name="jit(fn)/cholesky" stack_frame_id=4}
  %cholesky.1 = f64[1500,1500]{1,0} cholesky(%p0.1), lower=true, metadata={op_name="jit(fn)/cholesky" stack_frame_id=6}
  %fft.0 = c128[512,257,1500]{2,1,0} fft(%p0.1), fft_type=RFFT, metadata={op_name="jit(fn)/fft" stack_frame_id=8}
  %fft.1 = c128[512,257]{1,0} fft(%p0.1), fft_type=RFFT, metadata={op_name="jit(fn)/fft" stack_frame_id=10}
  %add.5 = f64[1500,1500]{1,0} add(%p0.1, %p0.1), metadata={op_name="jit(fn)/add" stack_frame_id=1}
  %add.11 = f64[1500,3]{1,0} add(%p0.1, %p0.1), metadata={op_name="jit(fn)/add" stack_frame_id=13}
  ROOT %reduce.0 = f64[] reduce(%cholesky.0), dimensions={0,1}, metadata={op_name="jit(fn)/reduce_sum" stack_frame_id=4}
}
"""


@pytest.fixture
def stack_index() -> xa.StackFrameIndex:
    return xa.StackFrameIndex.from_module_proto(MODULE_PROTO_BLOB)


@pytest.fixture
def index(stack_index) -> dict[str, xa.Instruction]:
    return xa.hlo_index(FIXTURE_HLO, stack_index)


# ---------------------------------------------------------------------------
# The stack frame index
# ---------------------------------------------------------------------------


def test_stack_frame_index_round_trips_from_a_module_proto(stack_index):
    """Field 17 of the module proto is the stack frame index, and it decodes."""
    assert stack_index.file_names == _FILES
    assert stack_index.function_names == _FUNCTIONS
    assert len(stack_index.stack_frames) == len(_STACK_FRAMES)


def test_frames_are_innermost_first_and_walk_the_parent_chain(stack_index):
    """Frame 4 is the Cholesky *called by* log_det_curvature_reg_matrix_term."""
    frames = stack_index.frames_for(4)
    assert len(frames) == 2
    assert frames[0].line == 872
    assert frames[0].function == "_log_det_symmetric_from"
    assert frames[1].line == 894
    assert frames[1].function == "log_det_curvature_reg_matrix_term"


def test_an_unknown_stack_frame_id_is_an_empty_stack_not_an_exception(stack_index):
    assert stack_index.frames_for(0) == ()
    assert stack_index.frames_for(9999) == ()


def test_a_module_proto_without_a_stack_frame_index_is_empty_not_an_error():
    empty = xa.StackFrameIndex.from_module_proto(_len_field(1, b"jit_nothing"))
    assert empty.file_names == []
    assert empty.frames_for(1) == ()


# ---------------------------------------------------------------------------
# HLO parsing
# ---------------------------------------------------------------------------


def test_hlo_index_parses_every_instruction_including_fused_computations(index):
    for name in (
        "fusion.mixed",
        "fusion.walk",
        "cholesky.0",
        "cholesky.1",
        "fft.0",
        "fft.1",
        "add.5",
        "reduce.0",
        "add.7",  # inside %fused_computation.mixed
        "gather.9",  # inside %fused_computation.mixed
        "select.2",  # inside %fused_computation.walk
    ):
        assert name in index, f"{name} missing from the HLO index"


def test_opcode_shape_and_metadata_are_read_off_each_instruction(index):
    add = index["add.5"]
    assert add.opcode == "add"
    assert add.dims == (1500, 1500)
    assert add.op_name == "jit(fn)/add"
    assert add.stack_frame_id == 1
    assert add.source == f"{_ABSTRACT}:371"

    fft = index["fft.0"]
    assert fft.opcode == "fft"
    assert fft.dims == (512, 257, 1500)


def test_fusions_record_the_computation_they_call(index):
    assert index["fusion.mixed"].calls == ("fused_computation.mixed",)
    assert index["fusion.walk"].calls == ("fused_computation.walk",)
    assert index["add.7"].computation == "fused_computation.mixed"
    assert index["add.5"].computation == "main.9"


def test_operands_are_parsed_so_a_consumer_can_be_walked_back(index):
    """The only route to ``F + lambda*H`` once XLA has fused the sum away."""
    assert index["cholesky.0"].operands == ("fusion.mixed",)
    assert index["fusion.mixed"].operands == ("p0.1", "p0.1")
    assert index["add.7"].operands == ("param_0.1", "param_1.1")
    assert index["p0.1"].operands == ()


def test_resolve_producer_walks_through_bitcasts_and_copies(index):
    """Otherwise every producer in a GPU module is "a bitcast"."""
    text = FIXTURE_HLO.replace(
        "%cholesky.0 = f64[1400,1400]{1,0} cholesky(%fusion.mixed), lower=true,",
        "%bitcast.9 = f64[1400,1400]{0,1} bitcast(%fusion.mixed)\n"
        "  %copy.9 = f64[1400,1400]{0,1} copy(%bitcast.9)\n"
        "  %cholesky.0 = f64[1400,1400]{1,0} cholesky(%copy.9), lower=true,",
    )
    idx = xa.hlo_index(text, xa.StackFrameIndex.from_module_proto(MODULE_PROTO_BLOB))
    assert idx["cholesky.0"].operands == ("copy.9",)
    assert xa.resolve_producer(idx, "copy.9").name == "fusion.mixed"


def test_resolve_producer_crosses_a_fusion_boundary_by_parameter_index():
    """A fused computation's parameter N is the calling fusion's operand N.

    Without that hop, a consumer inside a fusion reports its producer as
    ``param_0`` — which names nothing.
    """
    text = """\
HloModule m, entry_computation_layout={(f64[4,4]{1,0})->f64[4,4]{1,0}}

%fused_inner (param_0.9: f64[4,4], param_1.9: f64[4,4]) -> f64[4,4] {
  %param_0.9 = f64[4,4]{1,0} parameter(0)
  %param_1.9 = f64[4,4]{1,0} parameter(1)
  ROOT %add.9 = f64[4,4]{1,0} add(%param_0.9, %param_1.9)
}

ENTRY %main.3 (x.1: f64[4,4]) -> f64[4,4] {
  %x.1 = f64[4,4]{1,0} parameter(0)
  %real_work.1 = f64[4,4]{1,0} dot(%x.1, %x.1)
  ROOT %the_fusion = f64[4,4]{1,0} fusion(%x.1, %real_work.1), kind=kLoop, calls=%fused_inner
}
"""
    idx = xa.hlo_index(text)
    assert idx["the_fusion"].operands == ("x.1", "real_work.1")
    assert idx["param_1.9"].parameter_index == 1
    # param_1.9 is operand 1 of the fusion, i.e. the dot -- not the entry parameter.
    assert xa.resolve_producer(idx, "param_1.9").name == "real_work.1"
    assert xa.resolve_producer(idx, "param_0.9").name == "x.1"


def test_a_parameter_of_a_computation_with_several_callers_is_not_guessed():
    """Several callers means several origins; naming one would be a guess."""
    text = """\
HloModule m, entry_computation_layout={(f64[4,4]{1,0})->f64[4,4]{1,0}}

%shared (param_0.9: f64[4,4]) -> f64[4,4] {
  ROOT %param_0.9 = f64[4,4]{1,0} parameter(0)
}

ENTRY %main.3 (x.1: f64[4,4]) -> f64[4,4] {
  %x.1 = f64[4,4]{1,0} parameter(0)
  %other.1 = f64[4,4]{1,0} dot(%x.1, %x.1)
  %f1 = f64[4,4]{1,0} fusion(%x.1), kind=kLoop, calls=%shared
  ROOT %f2 = f64[4,4]{1,0} fusion(%other.1), kind=kLoop, calls=%shared
}
"""
    idx = xa.hlo_index(text)
    assert xa.resolve_producer(idx, "param_0.9").name == "param_0.9"


def test_the_census_locates_the_sum_by_its_consumers(index):
    """``cholesky.0`` (the log-det reached from :894) consumes ``fusion.mixed``.

    ``fusion.mixed`` is the kernel whose body holds the ``add`` at
    ``abstract.py:371`` — so walking one operand back from the consumer finds
    the materialised matrix even though no standalone add survives.
    """
    where = xa.hlo_census(index)["curvature_reg_add_nn"]["where_the_sum_lives"]
    rows = {(r["consumer"], r["producer"]) for r in where["consumers_of_F_plus_lambda_H"]}
    assert ("cholesky.0", "fusion.mixed") in rows
    assert where["producers"]["fusion.mixed"] == ["cholesky.0"]


def test_constituents_of_a_fusion_are_its_fused_computation_members(index):
    members = {i.name for i in xa.constituents_of(index["fusion.mixed"], index)}
    assert members == {"param_0.1", "param_1.1", "add.7", "gather.9"}
    assert xa.constituents_of(index["cholesky.0"], index) == []


# ---------------------------------------------------------------------------
# The stage map
# ---------------------------------------------------------------------------


def test_the_two_log_dets_are_told_apart_by_their_caller_alone(index):
    """The whole reason :class:`StageRule` has ``requires``.

    Both Choleskys have the *same* innermost frame (``_log_det_symmetric_from``,
    abstract.py:872). Nothing but the caller distinguishes them, so if this
    passes the map is reading the stack and not the innermost line.
    """
    curvature, _ = xa.stages_of_instruction(index["cholesky.0"], index)
    regularization, _ = xa.stages_of_instruction(index["cholesky.1"], index)
    assert curvature == "log_det_curvature_reg"
    assert regularization == "log_det_regularization"
    assert index["cholesky.0"].frames[0].line == index["cholesky.1"].frames[0].line


def test_the_two_psf_convolutions_are_told_apart_by_their_caller_alone(index):
    """Same shared FFT body (convolver.py:586), different caller."""
    mapping_matrix, _ = xa.stages_of_instruction(index["fft.0"], index)
    image, _ = xa.stages_of_instruction(index["fft.1"], index)
    assert mapping_matrix == "psf_convolution_mapping_matrix"
    assert image == "psf_convolution_image"


def test_a_fusion_spanning_two_stages_is_mixed_not_its_representative(index):
    """``fusion.mixed`` is named ``gather`` but contains the F + lambda*H add.

    Taking the representative ``op_name`` would file the whole kernel under
    ``edge_subset_gather`` and lose the add entirely.
    """
    label, members = xa.stages_of_instruction(index["fusion.mixed"], index)
    assert label == xa.MIXED_FUSION
    assert members == frozenset({"curvature_reg_add", "edge_subset_gather"})


def test_the_border_relocator_is_not_swallowed_by_the_generic_mesh_rule():
    """``inversion/mesh/`` is a prefix of ``inversion/mesh/border_relocator.py``.

    On the shakeout the generic rule sat first and filed the whole relocator
    under ``mesh_construction``, so ordering is asserted, not assumed.
    """
    frame = xa.Frame(
        file=_ARRAY + "inversion/mesh/border_relocator.py",
        function="ellipse_params_via_border_pca_from",
        line=242,
    )
    assert xa.stage_for_frames((frame,)) == "border_relocator"


def test_the_harness_certified_solver_has_its_own_stage():
    """It is this repo's code, not PyAutoArray's, and it is the solver row.

    Without a rule its kernels walk outward to ``abstract.py:642`` (the library
    call site) and land in ``reconstruction_scatter``, where the shakeout's
    5.2 ms solve was hiding.
    """
    frame = xa.Frame(
        file="/x/autolens_profiling/scripts/misc/likelihood_breakdown/active_set_steps.py",
        function="_masked_solve",
        line=418,
    )
    assert xa.stage_for_frames((frame,)) == "certified_active_set_solve"


def test_a_fusion_whose_constituents_share_one_stage_is_that_stage(index):
    label, members = xa.stages_of_instruction(index["fusion.walk"], index)
    assert label == "mesh_locate_walk"
    assert members == frozenset({"mesh_locate_walk"})


def test_an_empty_stack_is_other_and_never_raises():
    assert xa.stage_for_frames(()) == xa.OTHER


def test_stage_rules_match_on_path_fragments_not_bare_basenames():
    """``delaunay.py`` exists in more than one package; the rule must not match it."""
    decoy = xa.Frame(
        file=_ARRAY + "inversion/pixelization/mesh/delaunay.py", function="whatever", line=200
    )
    assert xa.stage_for_frames((decoy,)) not in ("mesh_locate_walk", "mesh_qhull_callback")


def test_every_stage_map_label_appears_in_stage_order():
    """A label the printer does not know about would silently sort to the end."""
    labels = {rule.label for rule in xa.STAGE_MAP}
    missing = labels - set(xa.STAGE_ORDER)
    assert not missing, f"labels absent from STAGE_ORDER: {sorted(missing)}"


def test_stage_map_provenance_is_serialisable_and_complete():
    provenance = xa.stage_map_provenance()
    assert len(provenance) == len(xa.STAGE_MAP)
    assert all(set(row) == {"label", "path", "lines", "requires", "function"} for row in provenance)


# ---------------------------------------------------------------------------
# Attribution arithmetic
# ---------------------------------------------------------------------------


def _event(hlo_op, start_ns, dur_ns, name=None, stream="Stream #14"):
    return {
        "plane": "/device:GPU:0",
        "stream": stream,
        "name": name or (hlo_op or "Memset 3"),
        "start_ns": start_ns,
        "dur_ns": dur_ns,
        "hlo_op": hlo_op,
        "hlo_module": "jit_likelihood",
        "program_id": 10,
    }


#: One synthetic call: 1.0 + 2.0 + 0.5 + 1.5 ms of kernels with a 1.0 ms gap
#: between the second and third (the host-callback fingerprint), spanning
#: 7.0 ms of device time.
def _one_call(offset_ns: int) -> list[dict]:
    return [
        _event("fusion.walk", offset_ns + 0, 1_000_000),
        _event("fft.0", offset_ns + 1_000_000, 2_000_000),
        # 1.0 ms gap here
        _event("cholesky.0", offset_ns + 4_000_000, 500_000),
        _event("cholesky.1", offset_ns + 4_500_000, 1_500_000),
        _event(None, offset_ns + 6_000_000, 0, name="Memset 3"),
    ]


def test_every_event_lands_in_exactly_one_row(index):
    """Totality: the rows partition the events, with nothing double-counted."""
    events = _one_call(0)
    result = xa.attribute(events, index, wall_ms=8.0, calls=1)
    rows_ms = sum(row["median_ms"] for row in result["per_stage_ms"].values())
    events_ms = sum(e["dur_ns"] for e in events) / 1e6
    assert rows_ms == pytest.approx(events_ms, rel=1e-12)


def test_kernels_land_in_the_rows_their_source_says(index):
    result = xa.attribute(_one_call(0), index, wall_ms=8.0, calls=1)
    stages = {k: v["median_ms"] for k, v in result["per_stage_ms"].items()}
    assert stages["mesh_locate_walk"] == pytest.approx(1.0)
    assert stages["psf_convolution_mapping_matrix"] == pytest.approx(2.0)
    assert stages["log_det_curvature_reg"] == pytest.approx(0.5)
    assert stages["log_det_regularization"] == pytest.approx(1.5)
    assert stages[xa.MEMSET] == pytest.approx(0.0)


def test_device_idle_is_the_measured_gap_and_sum_is_rows_plus_idle(index):
    result = xa.attribute(_one_call(0), index, wall_ms=8.0, calls=1)
    assert result["device_span_ms"] == pytest.approx(6.0)
    assert result["device_busy_ms"] == pytest.approx(5.0)
    assert result["device_idle_ms"] == pytest.approx(1.0)
    assert result["kernel_ms"] == pytest.approx(5.0)
    assert result["sum_ms"] == pytest.approx(6.0)


def test_the_wall_residual_is_reported_not_absorbed(index):
    """8.0 ms wall against a 6.0 ms device span leaves 2.0 ms of host time.

    A helper that folded that into an idle row would report 0 % reconciliation
    and hide two milliseconds of dispatch.
    """
    result = xa.attribute(_one_call(0), index, wall_ms=8.0, calls=1)
    assert result["host_outside_span_ms"] == pytest.approx(2.0)
    assert result["reconciliation_pct"] == pytest.approx(100.0 * (6.0 - 8.0) / 8.0)


def test_the_largest_idle_gap_names_the_kernels_either_side(index):
    """The qhull ``pure_callback`` is invisible on the stream; its gap is not."""
    result = xa.attribute(_one_call(0), index, wall_ms=8.0, calls=1)
    largest = result["largest_idle_gaps"][0]
    assert largest["gap_ms"] == pytest.approx(1.0)
    assert largest["after"] == "fft.0"
    assert largest["before"] == "cholesky.0"


def test_multiple_calls_are_split_and_reported_per_call(index):
    events = _one_call(0) + _one_call(10_000_000) + _one_call(20_000_000)
    result = xa.attribute(events, index, wall_ms=8.0, calls=3)
    assert result["call_blocks"] == 3
    assert result["call_split"].startswith("per-call")
    # Each call is identical, so the median is one call's value, not three.
    assert result["per_stage_ms"]["psf_convolution_mapping_matrix"]["median_ms"] == pytest.approx(
        2.0
    )
    assert result["per_stage_ms"]["psf_convolution_mapping_matrix"]["min_ms"] == pytest.approx(2.0)


def test_an_unsplittable_trace_falls_back_to_dividing_and_says_so(index):
    """Better a stated single block than three invented calls."""
    events = _one_call(0)
    result = xa.attribute(events, index, wall_ms=8.0, calls=3)
    assert result["call_blocks"] == 1
    assert result["call_split"].startswith("single-block")
    assert result["per_stage_ms"]["psf_convolution_mapping_matrix"]["median_ms"] == pytest.approx(
        2.0 / 3
    )


def test_an_hlo_op_outside_the_module_is_unjoined_and_alarms(index):
    """A ``command_buffer_N`` event is exactly this case, and must be loud."""
    events = _one_call(0) + [_event("command_buffer_1", 7_000_000, 3_000_000)]
    result = xa.attribute(events, index, wall_ms=11.0, calls=1)
    assert result["per_stage_ms"][xa.UNJOINED]["median_ms"] == pytest.approx(3.0)
    assert result["unjoined_ms"] == pytest.approx(3.0)
    assert result["unjoined_pct"] > 0


def test_an_unmatched_source_is_other_and_keeps_its_file(index):
    """``other`` must say what the stage map missed, or it cannot be fixed."""
    unknown_index = dict(index)
    unknown_index["mystery.0"] = xa.Instruction(
        name="mystery.0",
        opcode="dot",
        computation="main.9",
        shape="f64[10,10]",
        frames=(xa.Frame(file="some/other/package/thing.py", function="f", line=12),),
    )
    result = xa.attribute([_event("mystery.0", 0, 4_000_000)], unknown_index, wall_ms=4.0, calls=1)
    assert result["per_stage_ms"][xa.OTHER]["median_ms"] == pytest.approx(4.0)
    assert "some/other/package/thing.py:12" in result["other_sources_ms"]


def test_mixed_fusion_records_its_constituent_stage_set(index):
    result = xa.attribute([_event("fusion.mixed", 0, 3_000_000)], index, wall_ms=3.0, calls=1)
    assert result["per_stage_ms"][xa.MIXED_FUSION]["median_ms"] == pytest.approx(3.0)
    sets = result["mixed_fusion"]["constituent_stage_sets"]
    assert sets == {"curvature_reg_add + edge_subset_gather": pytest.approx(3.0)}
    # The prorated view is an estimate and must never be in the measured table.
    assert result["mixed_fusion"]["prorated_estimate_ms"]["curvature_reg_add"] == pytest.approx(1.5)
    assert "ESTIMATE ONLY" in result["mixed_fusion"]["prorated_note"]


def test_each_stage_row_names_the_instructions_that_built_it(index):
    """A row nobody can check against the HLO is not a measurement."""
    result = xa.attribute(_one_call(0), index, wall_ms=8.0, calls=1)
    audit = result["stage_audit"]["psf_convolution_mapping_matrix"]
    assert [row["instruction"] for row in audit] == ["fft.0"]
    assert audit[0]["ms"] == pytest.approx(2.0)
    assert audit[0]["opcode"] == "fft"
    assert any("operated_mapping_matrix_list" in frame for frame in audit[0]["stack"])


def test_the_untraced_wall_is_recorded_beside_the_traced_one(index):
    """Reconciling a traced sum against an untraced wall hides profiler overhead.

    On the first real shakeout it made ``host_outside_span_ms`` negative, i.e.
    the device span outlasted the wall it was being reconciled against.
    """
    result = xa.attribute(_one_call(0), index, wall_ms=8.0, calls=1, untraced_wall_ms=6.5)
    assert result["untraced_wall_ms"] == pytest.approx(6.5)
    assert result["reconciliation_pct"] == pytest.approx(100.0 * (6.0 - 8.0) / 8.0)
    assert result["reconciliation_vs_untraced_pct"] == pytest.approx(100.0 * (6.0 - 6.5) / 6.5)
    assert result["host_outside_span_ms"] == pytest.approx(2.0)


def test_mixed_fusion_names_example_kernels_for_each_stage_set(index):
    result = xa.attribute([_event("fusion.mixed", 0, 3_000_000)], index, wall_ms=3.0, calls=1)
    assert result["mixed_fusion"]["example_kernels"] == {
        "curvature_reg_add + edge_subset_gather": ["fusion.mixed"]
    }


def test_calls_below_one_is_an_error_not_a_division(index):
    with pytest.raises(ValueError, match="calls must be >= 1"):
        xa.attribute(_one_call(0), index, wall_ms=8.0, calls=0)


def test_overlapping_events_on_two_streams_do_not_double_count_busy_time(index):
    """Union, not sum: two streams running concurrently are not 2x the span."""
    events = [
        _event("cholesky.0", 0, 2_000_000, stream="Stream #14"),
        _event("cholesky.1", 1_000_000, 2_000_000, stream="Stream #15"),
    ]
    result = xa.attribute(events, index, wall_ms=3.0, calls=1)
    assert result["device_span_ms"] == pytest.approx(3.0)
    assert result["device_busy_ms"] == pytest.approx(3.0)
    assert result["device_idle_ms"] == pytest.approx(0.0)
    # The kernel rows still sum to the raw 4.0 ms of kernel time.
    assert result["kernel_ms"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# The HLO census
# ---------------------------------------------------------------------------


def test_census_counts_the_curvature_reg_adds_and_keeps_their_callers(index):
    census = xa.hlo_census(index)
    adds = census["curvature_reg_add_nn"]
    assert adds["count"] == 2
    assert {i["name"] for i in adds["instructions"]} == {"add.5", "add.7"}
    assert adds["computations"] == ["fused_computation.mixed", "main.9"]
    assert set(adds["callers"]) == {"add.5", "add.7"}


def test_an_add_merely_CALLED_from_line_371_is_not_counted_as_the_sum(index):
    """The bug that made the first real census meaningless.

    ``curvature_reg_matrix`` is ``self._xp.add(self.curvature_matrix,
    self.regularization_matrix)`` on ONE line, and Python evaluates both
    arguments *at that line*. So ``abstract.py:371`` appears in the stack of
    every instruction downstream of it. Scanning the whole stack counted 15
    "F + lambda*H adds" on the real module whose innermost frames were in
    ``delaunay.py``, ``regularization_util.py`` and ``mapper_util.py``.

    ``add.11`` is exactly that shape: written at ``delaunay.py:659``, called
    from ``abstract.py:371``. It must not be counted, and its presence in the
    index proves the fixture really does carry the ambiguity.
    """
    assert index["add.11"].frames[0].file.endswith("delaunay.py")
    assert index["add.11"].frames[0].line == 659
    assert any(f.line == 371 for f in index["add.11"].frames), "fixture lost the outer frame"

    counted = {i["name"] for i in xa.hlo_census(index)["curvature_reg_add_nn"]["instructions"]}
    assert "add.11" not in counted
    assert counted == {"add.5", "add.7"}


def test_the_curvature_verdict_is_read_off_computations_not_the_raw_count():
    """XLA tiles the sum, so counting adds cannot answer "once or twice".

    The real shakeout found **13** adds traced at ``abstract.py:371`` and **zero**
    of them square. A verdict keyed on the raw count would have read that as "the
    JAX path rebuilds F + lambda*H thirteen times"; keyed on distinct
    computations it reads it as one tiled sum and says to check the callers.
    """
    one = xa.Instruction(name="add.1", opcode="add", computation="fused.0", shape="f64[64,300]")
    two = xa.Instruction(name="add.2", opcode="add", computation="fused.0", shape="f64[64,300]")
    verdict = xa._curvature_verdict([one, two], {"add": 2})
    assert "ONE computation" in verdict
    assert "ONE F + lambda*H" in verdict

    split = xa.Instruction(name="add.3", opcode="add", computation="fused.1", shape="f64[64,300]")
    assert "distinct computations" in xa._curvature_verdict([one, split], {"add": 2})


def test_no_add_at_the_line_is_not_the_same_as_no_line():
    """Two different "zero"s, and only one of them is a broken line number.

    The RTX leg found ``abstract.py:371`` carrying exactly one instruction — a
    ``transpose`` — and no ``add``: XLA fused the F + lambda*H sum into its
    consumers, which IS the answer to the curvature draft. Reporting that as
    "check the line number" would have thrown the finding away.
    """
    fused_away = xa._curvature_verdict([], {"transpose": 1})
    assert "NO add survives" in fused_away
    assert "transpose x1" in fused_away
    assert "fused the F + lambda*H sum into its consumers" in fused_away

    line_moved = xa._curvature_verdict([], {})
    assert "line number has moved" in line_moved


def test_census_reports_square_adds_separately_from_every_add_at_the_line():
    """Both numbers, because XLA slices.

    ``count`` is every add traced at :371 — a fused constituent can carry a tile
    shape rather than (n,n), and filtering on (n,n) first reported *zero* adds on
    a shakeout whose mixed fusions plainly contained the sum. ``count_square_nn``
    is the strict (n,n) reading, kept beside it.
    """
    text = FIXTURE_HLO.replace(
        "%add.5 = f64[1500,1500]{1,0} add(%p0.1, %p0.1)",
        "%add.5 = f64[1500]{0} add(%p0.1, %p0.1)",
    )
    index = xa.hlo_index(text, xa.StackFrameIndex.from_module_proto(MODULE_PROTO_BLOB))
    census = xa.hlo_census(index)
    assert census["curvature_reg_add_nn"]["count"] == 2
    assert census["curvature_reg_add_nn"]["count_square_nn"] == 1
    assert census["curvature_reg_add_nn"]["square_instructions"] == ["add.7"]


def test_census_splits_the_ffts_by_which_convolution_called_them(index):
    census = xa.hlo_census(index)
    assert census["fft_convolver_total"]["count"] == 2
    assert [i["name"] for i in census["fft_mapping_matrix"]["instructions"]] == ["fft.0"]
    assert [i["name"] for i in census["fft_image"]["instructions"]] == ["fft.1"]


def test_census_counts_the_edge_subset_gather(index):
    census = xa.hlo_census(index)
    assert census["edge_subset_gathers_613"]["count"] == 1
    assert census["edge_subset_gathers_613"]["instructions"][0]["name"] == "gather.9"


def test_census_counts_both_choleskys(index):
    census = xa.hlo_census(index)
    assert census["cholesky"]["count"] == 2
    assert set(census["cholesky"]["callers"]) == {"cholesky.0", "cholesky.1"}


def test_census_counts_a_cusolver_custom_call_as_a_factorization():
    """On the GPU backend the factorization is a custom-call named ``cholesky.N``.

    Matching the opcode alone found ZERO factorizations in the real module.
    """
    text = FIXTURE_HLO.replace(
        "%cholesky.0 = f64[1400,1400]{1,0} cholesky(%fusion.mixed), lower=true,",
        "%cholesky.0 = (f64[1400,1400]{0,1}, s32[]) custom-call(%fusion.mixed), "
        'custom_call_target="cusolver_potrf",',
    )
    index = xa.hlo_index(text, xa.StackFrameIndex.from_module_proto(MODULE_PROTO_BLOB))
    assert index["cholesky.0"].opcode == "custom-call"
    assert xa.hlo_census(index)["cholesky"]["count"] == 2


def test_census_reports_how_many_instructions_resolved_to_source(index):
    census = xa.hlo_census(index)
    assert census["instructions_total"] == len(index)
    assert 0 < census["instructions_with_frames"] <= census["instructions_total"]


# ---------------------------------------------------------------------------
# End-to-end, on whatever backend is present (skipped when JAX is absent)
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax", reason="the trace reader needs jax; parsing tests do not")


def test_the_real_hlo_and_proto_of_a_toy_jit_parse_and_resolve(tmp_path):
    """The parser must survive a module XLA actually emitted, not just a fixture.

    Runs on whatever backend is available (CPU is fine — this asserts the HLO
    text and the stack frame index, not device timings).
    """
    import jax.numpy as jnp

    def _inner(x):
        return jnp.sin(x)

    def _fn(x):
        return jnp.sum(_inner(x @ x))

    compiled = jax.jit(_fn).lower(jnp.ones((8, 8))).compile()
    stack_index = xa.StackFrameIndex.from_module_proto(
        compiled.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()
    )
    index = xa.hlo_index(compiled.as_text(), stack_index)

    assert index, "no instructions parsed from a real compiled module"
    resolved = [i for i in index.values() if i.frames]
    assert resolved, "no instruction resolved to a source frame -- field 17 moved?"
    # This test file is the innermost frame of at least one instruction.
    assert any(f.file.endswith("test_xla_attribution.py") for i in resolved for f in i.frames)
    # Function names are QUALIFIED (``test_...<locals>._inner`` for a nested def),
    # so the innermost frame is matched by suffix, not equality.
    assert any(i.frames[0].function.endswith("._inner") for i in resolved), (
        "no instruction resolved to the innermost helper: "
        + str(sorted({i.frames[0].function for i in resolved}))
    )


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="the device timeline needs a GPU backend"
)
def test_a_gpu_trace_joins_to_the_compiled_text(tmp_path):
    """The join itself, on hardware, with command buffers off."""
    import jax.numpy as jnp

    compiled = (
        jax.jit(lambda x: jnp.sum(x @ x))
        .lower(jnp.ones((256, 256)))
        .compile(compiler_options={"xla_gpu_enable_command_buffer": ""})
    )
    x = jnp.ones((256, 256))
    compiled(x).block_until_ready()
    with jax.profiler.trace(str(tmp_path), create_perfetto_link=False, create_perfetto_trace=False):
        for _ in range(3):
            compiled(x).block_until_ready()

    events = xa.device_events(tmp_path)
    assert events, "no device events in the trace"
    index = xa.hlo_index(compiled.as_text())
    joined = sum(e["dur_ns"] for e in events if e["hlo_op"] in index)
    total = sum(e["dur_ns"] for e in events)
    assert joined / total > 0.95, "command buffers are on, or the join broke"

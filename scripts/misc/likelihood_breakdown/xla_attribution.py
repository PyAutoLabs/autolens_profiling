"""
Attribute XLA device kernels of ONE compiled program to library source stages.

Phase 1 of the ``hst-gpu-non-solver-residue`` epic (autolens_profiling#268).

Every JAX decomposition this repo owns before this module is ~11 *separately
compiled* prefix programs whose rows move with fusion. This module reads the
**fused production program** instead: it joins every GPU kernel on the device
timeline to the optimized HLO instruction that launched it, and every
instruction to the library source frames that traced it — so a per-stage
millisecond table is a measurement of the program production runs, not
arithmetic across compilations.

The three inputs
----------------

1. ``compiled.as_text()`` — the **optimized** HLO. Instruction names in this
   text are what the profiler reports as each kernel's ``hlo_op``.
2. ``compiled.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()``
   — the module proto, read only for its **stack frame index** (the file/line
   table the text's ``stack_frame_id=N`` points into).
3. ``jax.profiler.trace(log_dir)`` → ``<log_dir>/**/**.xplane.pb``, read with
   ``jax.profiler.ProfileData.from_file`` — the device timeline.

Command buffers must be OFF for the traced executable
-----------------------------------------------------

**This is not optional and it is the whole join.** With XLA's defaults the GPU
module is captured into a CUDA graph, and every kernel inside it reports
``hlo_op="command_buffer_1"`` instead of its own instruction name. Measured on
the spike (RTX 2060, jax 0.10.2, two matmuls + a Cholesky):

=========================================  =================  ==========================
``xla_gpu_enable_command_buffer``          distinct ``hlo_op``  device duration joined
=========================================  =================  ==========================
default (on)                               6                  **23.0 %**
``""`` (off)                               12                 **100.0 %**
=========================================  =================  ==========================

Kernel *names* do not rescue it: a cuBLAS GEMM is reported as
``void magma_sgemmEx_kernel<...>`` and a cuSOLVER Cholesky as
``volta_dgemm_64x64_lower_nt``, neither of which is an HLO name. Only
``hlo_op`` identifies them, and only with the graphs off.

Pass the flag **per compilation**, not through ``XLA_FLAGS``::

    ex_traced = lowered.compile(compiler_options={"xla_gpu_enable_command_buffer": ""})

so the same process can also hold the command-buffers-ON executable and time
both. The ON/OFF wall difference is launch overhead the graphs remove; it is a
property of the measurement and every leg records it rather than assuming it is
small (it was 30 % of the toy call on the RTX).

Source metadata is a stack, not a ``source_file=``
--------------------------------------------------

This JAX no longer prints ``source_file=``/``source_line=`` in HLO metadata. It
prints ``metadata={op_name="jit(f)/dot_general" stack_frame_id=11}`` and puts the
file/line table in ``HloModuleProto.stack_frame_index`` (field 17). That is
strictly *more* information: resolving a ``stack_frame_id`` and walking its
``parent_frame_id`` chain gives the full Python stack, innermost frame first.

This matters because the stages are not separable by the innermost frame alone.
Both Bayesian-evidence log determinants are the *same* three lines of
``inversion/inversion/abstract.py`` (``_log_det_symmetric_from``, 850-882) and are
told apart only by their caller — ``log_det_curvature_reg_matrix_term`` (894) or
``log_det_regularization_matrix_term`` (941). So a :class:`StageRule` may declare
a ``requires`` frame that must also appear somewhere in the same stack.

Fusions carry their constituents
--------------------------------

A fused kernel's entry instruction carries one representative ``op_name``, and
the ``%fused_computation*`` it ``calls=`` holds every constituent with its *own*
``op_name`` and ``stack_frame_id``. A fusion is therefore attributed from the
**set** of its constituents' stages. When that set has one member the kernel is
that stage; when it spans stages the kernel goes to ``mixed_fusion`` with the
constituent set recorded, rather than being assigned to whichever constituent
the representative metadata happened to name.

What the table sums to
----------------------

``attribute`` returns rows that are all measured:

- one row per stage, plus ``mixed_fusion``, ``other`` (joined but unmatched by
  every rule — the row that says what the stage map missed, with its source
  files kept), ``memset`` and ``unjoined`` (an event whose ``hlo_op`` is not an
  instruction of this module — with command buffers off this should be empty,
  and it is the join's own alarm);
- ``device_idle_ms`` — the union-complement of the kernel intervals inside the
  device span, i.e. real gaps on the stream;
- ``host_outside_span_ms`` — ``wall - device_span``, the dispatch and teardown
  that happens before the first kernel and after the last.

``sum_ms`` is the stage rows plus ``device_idle_ms``, and
``reconciliation_pct`` compares it to the wall. The residual is
``host_outside_span_ms`` and is reported, not absorbed.

No new dependency: the proto is read with a ~70-line wire-format reader below
(``xprof``/``tensorboard_plugin_profile``/``perfetto`` are not installed, and
importing TensorFlow's ``hlo_pb2`` beside a live JAX GPU backend was rejected).
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

# ===========================================================================
# PART 1 — a minimal protobuf wire reader, for the stack frame index only
# ===========================================================================
# Field numbers verified empirically (spike, 2026-09-16) against a module whose
# Python frames were known by construction, and re-verified by
# ``StackFrameIndex.from_module_proto`` refusing a blob whose field 17 does not
# decode to plausible file names.

#: ``HloModuleProto.stack_frame_index``.
_MODULE_STACK_FRAME_INDEX_FIELD = 17

_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LEN = 2
_WIRE_32BIT = 5


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    """Decode a base-128 varint from *buf* at *i*; return ``(value, next_i)``."""
    result = 0
    shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def iter_proto_fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield ``(field_number, wire_type, value)`` for every field in *buf*.

    ``value`` is an ``int`` for varint / fixed-width fields and ``bytes`` for
    length-delimited ones. Unknown fields are yielded like any other, which is
    what makes this forward-compatible with proto additions.
    """
    i = 0
    n = len(buf)
    while i < n:
        key, i = _read_varint(buf, i)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == _WIRE_VARINT:
            value, i = _read_varint(buf, i)
            yield field_number, wire_type, value
        elif wire_type == _WIRE_LEN:
            length, i = _read_varint(buf, i)
            if i + length > n:
                raise ValueError("truncated length-delimited field")
            yield field_number, wire_type, buf[i : i + length]
            i += length
        elif wire_type == _WIRE_64BIT:
            yield field_number, wire_type, int.from_bytes(buf[i : i + 8], "little")
            i += 8
        elif wire_type == _WIRE_32BIT:
            yield field_number, wire_type, int.from_bytes(buf[i : i + 4], "little")
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")


def _scalars(sub: bytes) -> dict[int, int]:
    """Varint fields of a small submessage, as ``{field_number: value}``."""
    return {f: v for f, w, v in iter_proto_fields(sub) if w == _WIRE_VARINT}


@dataclass(frozen=True)
class Frame:
    """One Python frame: where an HLO instruction was traced from."""

    #: Absolute path of the source file, as the tracing process saw it.
    file: str
    #: **Qualified** function name: a nested ``def`` appears as
    #: ``outer.<locals>.inner`` and a method as ``Class.method``, so
    #: :attr:`StageRule.function` must be written to match a qualname.
    function: str
    line: int
    column: int = 0

    @property
    def basename(self) -> str:
        return os.path.basename(self.file)

    def __str__(self) -> str:
        return f"{self.file}:{self.line} ({self.function})"


@dataclass
class StackFrameIndex:
    """``StackFrameIndexProto``: ``stack_frame_id`` -> a stack of :class:`Frame`.

    All ids in the proto are **1-based**, and ``parent_frame_id == 0`` means
    "no parent". :meth:`frames_for` returns the stack innermost frame first.
    """

    file_names: list[str] = field(default_factory=list)
    function_names: list[str] = field(default_factory=list)
    #: ``(file_name_id, function_name_id, line, column)``, 1-based ids.
    file_locations: list[tuple[int, int, int, int]] = field(default_factory=list)
    #: ``(file_location_id, parent_frame_id)``, 1-based ids.
    stack_frames: list[tuple[int, int]] = field(default_factory=list)

    # StackFrameIndexProto field numbers.
    _F_FILE_NAMES = 1
    _F_FUNCTION_NAMES = 2
    _F_FILE_LOCATIONS = 3
    _F_STACK_FRAMES = 4
    # FileLocation field numbers.
    _F_LOC_FILE = 1
    _F_LOC_FUNCTION = 2
    _F_LOC_LINE = 3
    _F_LOC_COLUMN = 4
    # StackFrame field numbers.
    _F_FRAME_LOCATION = 1
    _F_FRAME_PARENT = 2

    @classmethod
    def from_bytes(cls, blob: bytes) -> StackFrameIndex:
        """Parse a serialized ``StackFrameIndexProto``."""
        self = cls()
        for fno, wire, value in iter_proto_fields(blob):
            if wire != _WIRE_LEN:
                continue
            assert isinstance(value, bytes)
            if fno == cls._F_FILE_NAMES:
                self.file_names.append(value.decode("utf-8", "replace"))
            elif fno == cls._F_FUNCTION_NAMES:
                self.function_names.append(value.decode("utf-8", "replace"))
            elif fno == cls._F_FILE_LOCATIONS:
                s = _scalars(value)
                self.file_locations.append(
                    (
                        s.get(cls._F_LOC_FILE, 0),
                        s.get(cls._F_LOC_FUNCTION, 0),
                        s.get(cls._F_LOC_LINE, 0),
                        s.get(cls._F_LOC_COLUMN, 0),
                    )
                )
            elif fno == cls._F_STACK_FRAMES:
                s = _scalars(value)
                self.stack_frames.append(
                    (s.get(cls._F_FRAME_LOCATION, 0), s.get(cls._F_FRAME_PARENT, 0))
                )
        return self

    @classmethod
    def from_module_proto(cls, blob: bytes) -> StackFrameIndex:
        """Parse ``HloModuleProto`` and return its ``stack_frame_index``.

        An empty index (a module compiled without source info) is returned as an
        empty :class:`StackFrameIndex` rather than an error — the caller then
        gets ``other`` rows with empty frames, which is a visible failure.
        """
        for fno, wire, value in iter_proto_fields(blob):
            if fno == _MODULE_STACK_FRAME_INDEX_FIELD and wire == _WIRE_LEN:
                assert isinstance(value, bytes)
                return cls.from_bytes(value)
        return cls()

    def _location(self, loc_id: int) -> Frame | None:
        if not 1 <= loc_id <= len(self.file_locations):
            return None
        file_id, func_id, line, column = self.file_locations[loc_id - 1]
        file_name = self.file_names[file_id - 1] if 1 <= file_id <= len(self.file_names) else ""
        function = (
            self.function_names[func_id - 1] if 1 <= func_id <= len(self.function_names) else ""
        )
        return Frame(file=file_name, function=function, line=int(line), column=int(column))

    def frames_for(self, stack_frame_id: int) -> tuple[Frame, ...]:
        """The stack for *stack_frame_id*, **innermost frame first**."""
        frames: list[Frame] = []
        seen: set[int] = set()
        current = int(stack_frame_id)
        while 1 <= current <= len(self.stack_frames) and current not in seen:
            seen.add(current)
            loc_id, parent = self.stack_frames[current - 1]
            frame = self._location(loc_id)
            if frame is not None:
                frames.append(frame)
            current = int(parent)
        return tuple(frames)


# ===========================================================================
# PART 2 — the optimized HLO text index
# ===========================================================================

#: ``%name = shape opcode(operands), ...`` — the instruction line of HLO text.
_INSTRUCTION_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-$]+)\s*=\s*(?P<rest>\S.*)$",
)
#: The opcode is the first identifier after the shape, immediately before ``(``.
_OPCODE_RE = re.compile(r"(?P<shape>.*?)\b(?P<opcode>[a-z][\w\-]*)\(", re.S)
_METADATA_RE = re.compile(r"metadata=\{(?P<body>[^}]*)\}")
_OP_NAME_RE = re.compile(r'op_name="(?P<value>[^"]*)"')
_OP_TYPE_RE = re.compile(r'op_type="(?P<value>[^"]*)"')
_STACK_FRAME_ID_RE = re.compile(r"stack_frame_id=(?P<value>\d+)")
_CALLS_RE = re.compile(r"calls=(?P<value>[%\w.\-,\s]+?)(?:,\s*\w+=|\s*$)")


def _operand_names(rest: str, start: int) -> tuple[str, ...]:
    """Operand names from the parenthesised list beginning at *start* in *rest*.

    Scanned with a depth counter rather than a regex: an operand can itself carry
    brackets (``f64[1500,1500]{1,0} %x``) and a tuple-shaped custom-call's operand
    list contains nested parentheses, both of which a flat ``[^()]*`` match
    truncates. Only ``%``-prefixed tokens are operands; everything else in the
    list is shape or layout text.
    """
    depth = 0
    end = start
    for index, char in enumerate(rest[start:], start=start):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                end = index
                break
    else:
        return ()

    inner = rest[start + 1 : end]
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in inner:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))

    names: list[str] = []
    for part in parts:
        token = part.strip()
        if "%" not in token:
            continue
        name = token[token.index("%") + 1 :].strip()
        name = re.split(r"[\s,()\[\]{}]", name)[0]
        if name:
            names.append(name)
    return tuple(names)


_COMPUTATION_RE = re.compile(r"^\s*(?:ENTRY\s+)?%?(?P<name>[\w.\-$]+)\s*\(.*\)\s*->.*\{\s*$")
#: Leading shape of an instruction, e.g. ``f64[1500,1500]{1,0}``.
_SHAPE_RE = re.compile(r"^(?P<dtype>[a-z]\d*)\[(?P<dims>[\d,]*)\]")


@dataclass
class Instruction:
    """One optimized-HLO instruction, with its source stack resolved."""

    name: str
    opcode: str
    computation: str
    shape: str = ""
    op_name: str = ""
    op_type: str = ""
    stack_frame_id: int = 0
    frames: tuple[Frame, ...] = ()
    #: Computations this instruction ``calls=`` (fusions, while bodies, reduces).
    calls: tuple[str, ...] = ()
    #: Names of the instructions this one consumes, in operand order. Kept so a
    #: census can walk back from a consumer to the value it consumed — which is
    #: the only way to locate ``F + lambda*H`` once XLA has fused the sum away.
    operands: tuple[str, ...] = ()
    #: For a ``parameter`` instruction, its position in its computation's
    #: signature — the hop that crosses a fusion boundary outward.
    parameter_index: int | None = None

    @property
    def dims(self) -> tuple[int, ...]:
        """Dimensions of the instruction's output shape, ``()`` when unparsable."""
        match = _SHAPE_RE.match(self.shape.strip())
        if match is None:
            return ()
        dims = match.group("dims")
        return tuple(int(d) for d in dims.split(",") if d.strip()) if dims else ()

    @property
    def source(self) -> str:
        """``file:line`` of the innermost frame, ``""`` when unresolved."""
        return f"{self.frames[0].file}:{self.frames[0].line}" if self.frames else ""


def parse_computations(text: str) -> dict[str, list[str]]:
    """Split HLO *text* into ``{computation_name: [instruction lines]}``.

    Brace depth is tracked rather than regexed so a fused computation nested in
    another computation's body is attributed to the computation it is declared
    in, which is what ``calls=`` resolution needs.
    """
    computations: dict[str, list[str]] = {}
    stack: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        header = _COMPUTATION_RE.match(raw)
        if header is not None and stripped.endswith("{"):
            name = header.group("name")
            stack.append(name)
            computations.setdefault(name, [])
            continue
        if stripped.startswith("}"):
            if stack:
                stack.pop()
            continue
        if stack:
            computations[stack[-1]].append(raw)
    return computations


def _parse_instruction(line: str, computation: str) -> Instruction | None:
    match = _INSTRUCTION_RE.match(line)
    if match is None:
        return None
    name = match.group("name")
    rest = match.group("rest")
    opcode_match = _OPCODE_RE.match(rest)
    if opcode_match is None:
        return None
    opcode = opcode_match.group("opcode")
    shape = opcode_match.group("shape").strip()

    op_name = op_type = ""
    stack_frame_id = 0
    metadata = _METADATA_RE.search(rest)
    if metadata is not None:
        body = metadata.group("body")
        if (m := _OP_NAME_RE.search(body)) is not None:
            op_name = m.group("value")
        if (m := _OP_TYPE_RE.search(body)) is not None:
            op_type = m.group("value")
        if (m := _STACK_FRAME_ID_RE.search(body)) is not None:
            stack_frame_id = int(m.group("value"))

    operands = _operand_names(rest, opcode_match.end() - 1)

    parameter_index = None
    if opcode == "parameter":
        digits = re.match(r"\((\d+)\)", rest[opcode_match.end() - 1 :])
        if digits is not None:
            parameter_index = int(digits.group(1))

    calls: tuple[str, ...] = ()
    if (m := _CALLS_RE.search(rest)) is not None:
        calls = tuple(
            part.strip().lstrip("%") for part in m.group("value").split(",") if part.strip()
        )

    return Instruction(
        name=name,
        opcode=opcode,
        computation=computation,
        shape=shape,
        op_name=op_name,
        op_type=op_type,
        stack_frame_id=stack_frame_id,
        calls=calls,
        operands=operands,
        parameter_index=parameter_index,
    )


def hlo_index(text: str, stack_index: StackFrameIndex | None = None) -> dict[str, Instruction]:
    """Index every instruction in optimized HLO *text* by its name.

    Instruction names are unique across the module (XLA uniquifies them), so a
    flat name -> :class:`Instruction` map is enough to resolve both the
    profiler's ``hlo_op`` and any ``calls=`` target's members.
    """
    index: dict[str, Instruction] = {}
    for computation, lines in parse_computations(text).items():
        for line in lines:
            instruction = _parse_instruction(line, computation)
            if instruction is None:
                continue
            if stack_index is not None and instruction.stack_frame_id:
                instruction.frames = stack_index.frames_for(instruction.stack_frame_id)
            index[instruction.name] = instruction
    return index


def constituents_of(
    instruction: Instruction,
    index: Mapping[str, Instruction],
    computations: Mapping[str, list[str]] | None = None,
    _seen: set[str] | None = None,
) -> list[Instruction]:
    """Every instruction inside the computations *instruction* ``calls=``.

    Recurses through nested calls (a fusion whose body contains a reduce, a
    ``while`` whose body is itself fused) and returns the leaves plus every
    intermediate, each with its own frames. ``[]`` for a non-calling
    instruction.
    """
    if _seen is None:
        _seen = set()
    out: list[Instruction] = []
    for name in instruction.calls:
        if name in _seen:
            continue
        _seen.add(name)
        for candidate in index.values():
            if candidate.computation != name:
                continue
            out.append(candidate)
            if candidate.calls:
                out.extend(constituents_of(candidate, index, computations, _seen))
    return out


# ===========================================================================
# PART 3 — the stage map
# ===========================================================================


@dataclass(frozen=True)
class StageRule:
    """One ordered rule: a source-file fragment, optionally narrowed by lines.

    ``path`` is matched as a substring of a frame's file path with ``/``
    separators, so ``"inversion/mesh/interpolator/delaunay.py"`` cannot be
    satisfied by ``inversion/pixelization/delaunay.py``.

    ``lines`` is an inclusive ``(first, last)`` range on the frame's line.

    ``requires`` is a second ``(path, lines)`` pair that must match **some other
    frame of the same stack**. It exists because the two Bayesian log
    determinants are literally the same three lines of the same function and
    differ only in their caller; nothing else can tell them apart, and guessing
    from the representative ``op_name`` would silently merge them.

    ``function`` further narrows on the frame's Python function name.
    """

    label: str
    path: str
    lines: tuple[int, int] | None = None
    requires: tuple[str, tuple[int, int] | None] | None = None
    function: str | None = None

    def matches_frame(self, frame: Frame) -> bool:
        if self.path not in frame.file.replace("\\", "/"):
            return False
        if self.lines is not None and not self.lines[0] <= frame.line <= self.lines[1]:
            return False
        if self.function is not None and self.function != frame.function:
            return False
        return True

    def satisfied_by(self, frames: Sequence[Frame]) -> bool:
        if self.requires is None:
            return True
        req_path, req_lines = self.requires
        for frame in frames:
            if req_path not in frame.file.replace("\\", "/"):
                continue
            if req_lines is not None and not req_lines[0] <= frame.line <= req_lines[1]:
                continue
            return True
        return False


#: Label used for an instruction no rule matched. Its source files are kept in
#: the ``other_sources`` census so a run says what the map missed.
OTHER = "other"
#: Label for a kernel whose fused constituents span more than one stage.
MIXED_FUSION = "mixed_fusion"
#: Label for a device event whose ``hlo_op`` is not an instruction of the module
#: (with command buffers off this must be empty — it is the join's own alarm).
UNJOINED = "unjoined"
#: Memsets and other events the profiler emits without an ``hlo_op``.
MEMSET = "memset"

_ARRAY = "PyAutoArray/autoarray/"
_LENS = "PyAutoLens/autolens/"
_GALAXY = "PyAutoGalaxy/autogalaxy/"

_INVERSION_ABSTRACT = "inversion/inversion/abstract.py"
_DELAUNAY_INTERP = "inversion/mesh/interpolator/delaunay.py"
_CONVOLVER = "operators/convolver.py"
_IMAGING_ABSTRACT = "inversion/inversion/imaging/abstract.py"
_IMAGING_MAPPING = "inversion/inversion/imaging/mapping.py"
#: The phase-3 candidate convolutions (autolens_profiling#295) — harness code.
_PSF_CUBE_HARNESS = "likelihood_breakdown/psf_cube_injection.py"

#: Ordered rules. **Order is the contract**: a stack is walked innermost frame
#: first, and at each frame the rules are tried in this order, so the earliest
#: rule that matches the innermost identifying frame wins. Rules that key on a
#: *container* (``fit_imaging.py``, ``tracer.py``) therefore sit at the bottom —
#: they appear as outer frames on almost every instruction and would otherwise
#: swallow the inner stages.
#:
#: Line numbers are read from the installed PyAutoArray / PyAutoLens ``main``
#: (checked 2026-09-16) and are recorded in each leg's JSON under
#: ``stage_map_provenance`` so a later library edit that moves them shows up as
#: a growing ``other`` row rather than as a silently wrong table.
STAGE_MAP: tuple[StageRule, ...] = (
    # --- the two log determinants: same lines, told apart by their caller ----
    StageRule(
        "log_det_curvature_reg",
        _ARRAY + _INVERSION_ABSTRACT,
        (850, 882),
        requires=(_ARRAY + _INVERSION_ABSTRACT, (884, 896)),
    ),
    StageRule(
        "log_det_regularization",
        _ARRAY + _INVERSION_ABSTRACT,
        (850, 882),
        requires=(_ARRAY + _INVERSION_ABSTRACT, (897, 943)),
    ),
    # --- the F + lambda*H add, and the two [ids][:, ids] gathers -------------
    StageRule("curvature_reg_add", _ARRAY + _INVERSION_ABSTRACT, (359, 372)),
    StageRule("curvature_reg_reduce_gather", _ARRAY + _INVERSION_ABSTRACT, (374, 398)),
    # ``reconstruction`` spans 583-659. 606-614 is the edge-zeroed branch
    # (``ids_to_keep is not None``): the two ``[ids_to_keep]`` subsets. 615-640
    # is the partial solve plus the scatter back to full shape; 641-659 is the
    # else branch, which is what the Delaunay family runs (no edge zeroing).
    StageRule("edge_subset_gather", _ARRAY + _INVERSION_ABSTRACT, (606, 614)),
    StageRule("regularization_term", _ARRAY + _INVERSION_ABSTRACT, (786, 849)),
    StageRule("mapped_reconstruction", _ARRAY + _INVERSION_ABSTRACT, (722, 785)),
    StageRule("reconstruction_scatter", _ARRAY + _INVERSION_ABSTRACT, (583, 605)),
    StageRule("reconstruction_scatter", _ARRAY + _INVERSION_ABSTRACT, (615, 721)),
    # --- the PDIP / active-set solve ----------------------------------------
    # ``inversion_util.py`` ranges, read from the installed PyAutoArray (2026-09-16):
    # 12-95 curvature diag/added/mirrored helpers, 96-155 curvature_matrix_via_
    # mapping_matrix_from (F), 156-200 mapped_reconstructed_data_* (the mapped
    # reconstruction, NOT D), 201-255 reconstruction_positive_negative_from,
    # 256-488 reconstruction_positive_only_from (PDIP).
    # The certified active set is HARNESS code (``library_solver_injection``
    # rebinds the library's positive-only entry point), so its kernels trace to
    # this repo, not to PyAutoArray. Without these rules they walk outward to
    # ``abstract.py:642`` (the call site) and land in ``reconstruction_scatter``
    # — which is where the shakeout's 5.2 ms solve was hiding.
    StageRule("certified_active_set_solve", "likelihood_breakdown/active_set_steps.py"),
    StageRule("certified_active_set_solve", "likelihood_breakdown/library_solver_injection.py"),
    StageRule("pdip_solve", _ARRAY + "util/jax_nnls.py"),
    StageRule("pdip_solve", _ARRAY + "inversion/inversion/inversion_util.py", (256, 488)),
    StageRule(
        "mapped_reconstruction", _ARRAY + "inversion/inversion/inversion_util.py", (156, 200)
    ),
    StageRule("curvature_matrix_F", _ARRAY + "inversion/inversion/inversion_util.py", (12, 155)),
    StageRule(
        "reconstruction_scatter", _ARRAY + "inversion/inversion/inversion_util.py", (201, 255)
    ),
    # --- D and F, at their library entry points ------------------------------
    StageRule("data_vector_D", _ARRAY + _IMAGING_MAPPING, (54, 73)),
    StageRule("curvature_matrix_F", _ARRAY + _IMAGING_MAPPING, (74, 110)),
    StageRule(
        "data_vector_D",
        _ARRAY + "inversion/inversion/imaging/inversion_imaging_util.py",
        function="data_vector_via_blurred_mapping_matrix_from",
    ),
    StageRule(
        "curvature_matrix_F", _ARRAY + "inversion/inversion/imaging/inversion_imaging_util.py"
    ),
    # --- PSF convolution: the image, and the mapping-matrix cube -------------
    # The FFT body (convolver.py 555-607) is shared; the caller says which of the
    # two convolutions it is, exactly as the log dets do.
    # ``Convolver.convolved_mapping_matrix_from`` (1086-1300) convolves the
    # mapping-matrix CUBE; ``convolved_image_from`` / the over-sampled image
    # helpers (520-700, 1000-1085) convolve the 2D image. Both reach the same
    # rfft2/irfft2, so which one it is comes from the caller, exactly as the two
    # log determinants do.
    # Phase 3 (#295): the candidate convolutions are HARNESS code
    # (``psf_cube_injection`` rebinds ``Convolver.convolved_mapping_matrix_from``),
    # so the scatter / FFT / conv kernels a candidate writes trace to this repo.
    # Placed before the convolver rules; a candidate that calls back into
    # convolver.py (``real_space_direct``, ``mp_cube_c64``) has its innermost frame
    # in convolver.py and resolves there through the ``requires`` rules below.
    StageRule("psf_convolution_mapping_matrix", _PSF_CUBE_HARNESS),
    StageRule("psf_convolution_mapping_matrix", _ARRAY + _CONVOLVER, (1086, 1300)),
    StageRule("psf_convolution_mapping_matrix", _ARRAY + _CONVOLVER, (632, 642)),
    StageRule("psf_convolution_mapping_matrix", _ARRAY + _CONVOLVER, (670, 691)),
    StageRule(
        "psf_convolution_mapping_matrix",
        _ARRAY + _CONVOLVER,
        requires=(_ARRAY + _CONVOLVER, (1086, 1300)),
    ),
    StageRule(
        "psf_convolution_mapping_matrix",
        _ARRAY + _CONVOLVER,
        requires=(_ARRAY + _IMAGING_ABSTRACT, (119, 150)),
    ),
    StageRule("psf_convolution_image", _ARRAY + _CONVOLVER),
    StageRule("psf_convolution_mapping_matrix", _ARRAY + _IMAGING_ABSTRACT, (119, 150)),
    # --- mesh: the qhull callback, the walk, the weights ---------------------
    StageRule("mesh_qhull_callback", _ARRAY + _DELAUNAY_INTERP, (139, 172)),
    StageRule("mesh_seed_argmin", _ARRAY + _DELAUNAY_INTERP, (173, 207)),
    StageRule("mesh_locate_walk", _ARRAY + _DELAUNAY_INTERP, (208, 372)),
    StageRule("mesh_dual_areas", _ARRAY + _DELAUNAY_INTERP, (373, 396)),
    StageRule("mesh_dual_areas", _ARRAY + _DELAUNAY_INTERP, (437, 494)),
    StageRule("mesh_dual_areas", _ARRAY + _DELAUNAY_INTERP, (605, 615)),
    StageRule("mesh_interpolation_weights", _ARRAY + _DELAUNAY_INTERP, (616, 695)),
    StageRule("mesh_interpolation_weights", _ARRAY + _DELAUNAY_INTERP, (896, 1000)),
    StageRule("mesh_construction", _ARRAY + _DELAUNAY_INTERP),
    # --- border relocator (cells force it on; the library default is off) ----
    # BEFORE the generic ``inversion/mesh/`` rule: ``inversion/mesh/`` is a
    # prefix of ``inversion/mesh/border_relocator.py``, so the generic rule
    # swallowed the whole relocator into ``mesh_construction`` on the shakeout.
    StageRule("border_relocator", _ARRAY + "inversion/mesh/border_relocator.py"),
    StageRule("mesh_construction", _ARRAY + "inversion/mesh/"),
    # --- mapper: the mapping matrix ------------------------------------------
    StageRule("mapping_matrix", _ARRAY + "inversion/mappers/"),
    StageRule("mapping_matrix", _ARRAY + "inversion/mappings/"),
    StageRule("mapping_matrix", _ARRAY + "inversion/linear_obj/"),
    # --- regularization matrix H ---------------------------------------------
    StageRule("regularization_H", _ARRAY + "inversion/regularization/"),
    # --- chi squared, evidence -----------------------------------------------
    StageRule("chi_squared_evidence", _ARRAY + "fit/fit_util.py"),
    StageRule("chi_squared_evidence", _ARRAY + "fit/fit_dataset.py"),
    # --- ray tracing ----------------------------------------------------------
    StageRule("ray_trace_mesh_grid", _LENS + "lens/to_inversion.py"),
    StageRule("ray_trace_mesh_grid", _GALAXY + "galaxy/to_inversion.py"),
    StageRule("mass_profile_deflections", _GALAXY + "profiles/mass/"),
    StageRule("light_profile_image", _GALAXY + "profiles/light/"),
    StageRule("light_profile_image", _GALAXY + "operate/"),
    StageRule("ray_trace_data_grid", _LENS + "lens/tracer.py"),
    StageRule("ray_trace_data_grid", _LENS + "lens/tracer_util.py"),
    # --- grids / structures ---------------------------------------------------
    StageRule("grids_and_structures", _ARRAY + "structures/"),
    StageRule("grids_and_structures", _ARRAY + "operators/over_sampling/"),
    StageRule("grids_and_structures", _ARRAY + "geometry/"),
    # --- containers, last: they are outer frames on nearly everything ---------
    StageRule("inversion_other", _ARRAY + "inversion/"),
    StageRule("fit_imaging_other", _LENS + "imaging/fit_imaging.py"),
    StageRule("fit_imaging_other", _GALAXY + "imaging/fit_imaging.py"),
    StageRule("fit_imaging_other", _ARRAY + "fit/"),
    StageRule("analysis_other", _LENS + "analysis/"),
    StageRule("analysis_other", _GALAXY + "analysis/"),
)

#: The order stage rows are printed and plotted in — roughly the order the
#: likelihood call executes them. Labels not listed keep STAGE_MAP order after.
STAGE_ORDER: tuple[str, ...] = (
    "ray_trace_data_grid",
    "mass_profile_deflections",
    "light_profile_image",
    "psf_convolution_image",
    "ray_trace_mesh_grid",
    "border_relocator",
    "mesh_qhull_callback",
    "mesh_seed_argmin",
    "mesh_locate_walk",
    "mesh_dual_areas",
    "mesh_interpolation_weights",
    "mesh_construction",
    "mapping_matrix",
    "psf_convolution_mapping_matrix",
    "data_vector_D",
    "curvature_matrix_F",
    "regularization_H",
    "curvature_reg_add",
    "curvature_reg_reduce_gather",
    "edge_subset_gather",
    "pdip_solve",
    "certified_active_set_solve",
    "log_det_curvature_reg",
    "log_det_regularization",
    "reconstruction_scatter",
    "mapped_reconstruction",
    "regularization_term",
    "chi_squared_evidence",
    "grids_and_structures",
    "inversion_other",
    "fit_imaging_other",
    "analysis_other",
    MIXED_FUSION,
    OTHER,
    MEMSET,
    UNJOINED,
)


def stage_for_frames(frames: Sequence[Frame], stage_map: Sequence[StageRule] = STAGE_MAP) -> str:
    """Stage label for one source stack, innermost frame first.

    Frames are walked innermost outward and the rules are tried in order at each
    frame, so the most specific *place in the library* wins over the most
    specific rule. ``OTHER`` when nothing matches (including an empty stack).
    """
    for frame in frames:
        for rule in stage_map:
            if rule.matches_frame(frame) and rule.satisfied_by(frames):
                return rule.label
    return OTHER


def stages_of_instruction(
    instruction: Instruction,
    index: Mapping[str, Instruction],
    stage_map: Sequence[StageRule] = STAGE_MAP,
) -> tuple[str, frozenset[str]]:
    """``(label, constituent_stage_set)`` for one instruction.

    A non-calling instruction is its own stage and the set has one member. A
    fusion is the stage set of its constituents: one member means the kernel is
    that stage, more than one means :data:`MIXED_FUSION`. A fusion whose
    constituents resolve to nothing but ``OTHER`` stays ``OTHER`` rather than
    becoming a mixed row, so the census still sees it.
    """
    members = constituents_of(instruction, index)
    if not members:
        return stage_for_frames(instruction.frames, stage_map), frozenset()

    stages = {stage_for_frames(m.frames, stage_map) for m in members if m.frames}
    real = stages - {OTHER}
    if not real:
        # Nothing inside resolved; fall back to the fusion's own metadata.
        return stage_for_frames(instruction.frames, stage_map), frozenset(stages)
    if len(real) == 1:
        return next(iter(real)), frozenset(stages)
    return MIXED_FUSION, frozenset(real)


# ===========================================================================
# PART 4 — the device timeline
# ===========================================================================


def find_xplane(log_dir: str | os.PathLike) -> str:
    """The newest ``*.xplane.pb`` written under *log_dir*."""
    matches = glob.glob(os.path.join(str(log_dir), "**", "*.xplane.pb"), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"no *.xplane.pb under {log_dir!r}; jax.profiler.trace wrote nothing "
            f"(a trace with no device activity, or a profiler that failed to start)"
        )
    return max(matches, key=os.path.getmtime)


def device_events(log_dir: str | os.PathLike) -> list[dict]:
    """Every device-plane event of the trace at *log_dir*, sorted by start time.

    ``jax`` is imported here and nowhere else in this module, so the HLO parsing
    and the stage map stay importable (and unit-testable) without it.
    """
    import jax  # noqa: PLC0415 — deliberately lazy; see docstring

    profile = jax.profiler.ProfileData.from_file(find_xplane(log_dir))
    events: list[dict] = []
    for plane in profile.planes:
        if "/device" not in plane.name:
            continue
        for line in plane.lines:
            for event in line.events:
                stats = dict(event.stats)
                events.append(
                    {
                        "plane": plane.name,
                        "stream": line.name,
                        "name": event.name,
                        "start_ns": int(event.start_ns),
                        "dur_ns": int(event.duration_ns),
                        "hlo_op": stats.get("hlo_op"),
                        "hlo_module": stats.get("hlo_module"),
                        "program_id": stats.get("program_id"),
                    }
                )
    events.sort(key=lambda e: e["start_ns"])
    return events


def host_events(log_dir: str | os.PathLike, name_fragment: str = "") -> list[dict]:
    """Host-plane events, optionally filtered to names containing *name_fragment*.

    Used to look for the qhull ``pure_callback``: a host round-trip is invisible
    on the device stream and shows up either as a named host event or (when the
    profiler does not name it) only as the device-idle gap the callback creates.
    """
    import jax  # noqa: PLC0415

    profile = jax.profiler.ProfileData.from_file(find_xplane(log_dir))
    events: list[dict] = []
    for plane in profile.planes:
        if "/host" not in plane.name:
            continue
        for line in plane.lines:
            for event in line.events:
                if name_fragment and name_fragment not in event.name:
                    continue
                events.append(
                    {
                        "plane": plane.name,
                        "line": line.name,
                        "name": event.name,
                        "start_ns": int(event.start_ns),
                        "dur_ns": int(event.duration_ns),
                    }
                )
    events.sort(key=lambda e: e["start_ns"])
    return events


def split_calls(events: Sequence[dict], expected_calls: int) -> tuple[list[list[dict]], str]:
    """Split a trace of *expected_calls* identical executions into per-call lists.

    Every call runs the same instruction sequence, so the first event's
    ``(hlo_op, name)`` recurring is a call boundary. Returns
    ``(calls, method)``; when the boundary count disagrees with
    *expected_calls* the whole trace is returned as one block and ``method``
    says so, which keeps a min/max spread honest instead of inventing calls.
    """
    if not events:
        return [], "empty"
    key = (events[0]["hlo_op"], events[0]["name"])
    calls: list[list[dict]] = []
    current: list[dict] = []
    for event in events:
        if (event["hlo_op"], event["name"]) == key and current:
            calls.append(current)
            current = []
        current.append(event)
    if current:
        calls.append(current)
    if len(calls) != expected_calls:
        return [
            list(events)
        ], f"single-block (found {len(calls)} boundaries, expected {expected_calls})"
    return calls, "per-call (first-instruction recurrence)"


def _union_busy_ns(events: Iterable[dict]) -> int:
    """Total length of the union of the event intervals (overlap-safe)."""
    intervals = sorted((e["start_ns"], e["start_ns"] + e["dur_ns"]) for e in events)
    busy = 0
    cur_start: int | None = None
    cur_end = 0
    for start, end in intervals:
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        if start > cur_end:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_start is not None:
        busy += cur_end - cur_start
    return busy


def idle_gaps(events: Sequence[dict]) -> list[dict]:
    """Gaps between consecutive kernel intervals, largest first.

    A large single gap inside one call is the fingerprint of a host round-trip
    (the qhull ``pure_callback``): the device has nothing to run while the host
    triangulates. The kernels either side are recorded so the gap can be placed.
    """
    intervals = sorted((e["start_ns"], e["start_ns"] + e["dur_ns"], e) for e in events)
    gaps: list[dict] = []
    cur_end: int | None = None
    previous: dict | None = None
    for start, end, event in intervals:
        if cur_end is not None and start > cur_end:
            gaps.append(
                {
                    "gap_ms": (start - cur_end) / 1e6,
                    "after": (previous or {}).get("hlo_op") or (previous or {}).get("name"),
                    "before": event.get("hlo_op") or event.get("name"),
                }
            )
        if cur_end is None or end > cur_end:
            cur_end = end
            previous = event
    gaps.sort(key=lambda g: g["gap_ms"], reverse=True)
    return gaps


# ===========================================================================
# PART 5 — attribution
# ===========================================================================


def attribute(
    events: Sequence[dict],
    index: Mapping[str, Instruction],
    *,
    wall_ms: float,
    calls: int,
    stage_map: Sequence[StageRule] = STAGE_MAP,
    untraced_wall_ms: float | None = None,
    top_instructions: int = 6,
) -> dict:
    """Per-stage milliseconds per call, reconciled against *wall_ms*.

    *events* is one trace of *calls* identical executions; *wall_ms* is the
    whole-call wall time of the **same** executable, measured untraced.

    *wall_ms* must be the wall of the **traced** run, because that is the run the
    kernel durations came from; reconciling a traced sum against an untraced wall
    builds the profiler's own overhead into the residual (it made
    ``host_outside_span_ms`` negative on the first shakeout). Pass the untraced
    wall as *untraced_wall_ms* and it is recorded beside it with its own
    percentage, so the production-comparable number is never lost.

    Each stage also carries its ``top_instructions``: the HLO instructions that
    contributed most of its milliseconds, with their opcode, shape and full
    source stack. A stage row is only as good as the rules that built it, and a
    row nobody can check against the HLO is not a measurement.

    Everything in the returned dict is measured. ``per_stage_ms`` is the median
    over the individual calls when they could be split (with ``min``/``max``),
    and the trace total divided by *calls* when they could not — ``call_split``
    says which. ``sum_ms`` is the stage rows plus ``device_idle_ms``;
    ``reconciliation_pct`` compares that to ``wall_ms`` and the residual is
    ``host_outside_span_ms``, which is reported rather than absorbed.
    """
    if calls < 1:
        raise ValueError(f"calls must be >= 1 (got {calls})")

    stage_cache: dict[str, tuple[str, frozenset[str]]] = {}
    other_sources: dict[str, float] = {}
    mixed_sets: dict[tuple[str, ...], float] = {}

    def classify(event: dict) -> tuple[str, frozenset[str]]:
        op = event.get("hlo_op")
        if op is None:
            return (MEMSET if "memset" in event["name"].lower() else UNJOINED), frozenset()
        if op not in index:
            return UNJOINED, frozenset()
        if op not in stage_cache:
            stage_cache[op] = stages_of_instruction(index[op], index, stage_map)
        return stage_cache[op]

    call_blocks, split_method = split_calls(events, calls)
    n_blocks = len(call_blocks)

    per_call_stage_ms: list[dict[str, float]] = []
    for block in call_blocks:
        totals: dict[str, float] = {}
        for event in block:
            label, _ = classify(event)
            totals[label] = totals.get(label, 0.0) + event["dur_ns"] / 1e6
        per_call_stage_ms.append(totals)

    # A single block holds every call, so divide; per-call blocks do not.
    divisor = calls if n_blocks == 1 else 1

    labels = sorted({label for totals in per_call_stage_ms for label in totals})

    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        n = len(ordered)
        if n == 0:
            return 0.0
        mid = n // 2
        return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])

    per_stage_ms: dict[str, dict[str, float]] = {}
    for label in labels:
        series = [totals.get(label, 0.0) / divisor for totals in per_call_stage_ms]
        per_stage_ms[label] = {
            "median_ms": _median(series),
            "min_ms": min(series),
            "max_ms": max(series),
        }

    # Per-instruction milliseconds, so every stage row can name what built it.
    instruction_ms: dict[str, float] = {}
    instruction_stage: dict[str, str] = {}
    for event in events:
        op = event.get("hlo_op")
        if op is None:
            continue
        instruction_ms[op] = instruction_ms.get(op, 0.0) + event["dur_ns"] / 1e6 / calls
        instruction_stage[op] = classify(event)[0]

    def _audit(op: str) -> dict:
        instruction = index.get(op)
        return {
            "instruction": op,
            "ms": instruction_ms.get(op, 0.0),
            "opcode": instruction.opcode if instruction else None,
            "shape": instruction.shape if instruction else None,
            "source": instruction.source if instruction else None,
            "stack": [str(f) for f in instruction.frames[:5]] if instruction else [],
        }

    by_stage: dict[str, list[str]] = {}
    for op, label in instruction_stage.items():
        by_stage.setdefault(label, []).append(op)
    stage_audit = {
        label: [_audit(op) for op in sorted(ops, key=lambda o: -instruction_ms.get(o, 0.0))][
            :top_instructions
        ]
        for label, ops in by_stage.items()
    }

    # Census material: what ``other`` actually was, and what each mixed fusion mixed.
    mixed_examples: dict[tuple[str, ...], list[str]] = {}
    for event in events:
        label, members = classify(event)
        ms = event["dur_ns"] / 1e6 / calls
        if label == MIXED_FUSION:
            mixed_key = tuple(sorted(members))
            seen_kernels = mixed_examples.setdefault(mixed_key, [])
            if event["hlo_op"] not in seen_kernels:
                seen_kernels.append(event["hlo_op"])
        if label == OTHER:
            op = event.get("hlo_op")
            instruction = index.get(op) if op else None
            key = (
                instruction.source if instruction and instruction.source else (op or event["name"])
            )
            other_sources[key] = other_sources.get(key, 0.0) + ms
        elif label == MIXED_FUSION:
            mixed_key = tuple(sorted(members))
            mixed_sets[mixed_key] = mixed_sets.get(mixed_key, 0.0) + ms

    # Device span, busy time and the idle between kernels, per call.
    span_ns = (
        max(e["start_ns"] + e["dur_ns"] for e in events) - min(e["start_ns"] for e in events)
        if events
        else 0
    )
    busy_ns = _union_busy_ns(events)
    device_span_ms = span_ns / 1e6 / calls
    device_busy_ms = busy_ns / 1e6 / calls
    device_idle_ms = max(device_span_ms - device_busy_ms, 0.0)

    kernel_ms = sum(row["median_ms"] for row in per_stage_ms.values())
    sum_ms = kernel_ms + device_idle_ms
    host_outside_span_ms = wall_ms - device_span_ms
    reconciliation_pct = 100.0 * (sum_ms - wall_ms) / wall_ms if wall_ms else float("nan")

    unjoined_ms = per_stage_ms.get(UNJOINED, {}).get("median_ms", 0.0)
    other_ms = per_stage_ms.get(OTHER, {}).get("median_ms", 0.0)

    gaps = idle_gaps(call_blocks[0]) if call_blocks else []

    # Prorating a mixed fusion across its constituents is an ESTIMATE and is
    # labelled as one everywhere it appears. It exists so phase 2 can rank
    # levers; it is never the measured table and never reconciles anything.
    prorated: dict[str, float] = {}
    for members_key, ms in mixed_sets.items():
        if not members_key:
            continue
        share = ms / len(members_key)
        for label in members_key:
            prorated[label] = prorated.get(label, 0.0) + share

    return {
        "parser": "ProfileData.from_file(xplane) + as_text() + stack_frame_index(proto field 17)",
        "stage_audit": stage_audit,
        "stage_audit_note": (
            "The HLO instructions that contributed most of each stage's milliseconds, with "
            "their full source stack. A stage row is only as good as the rules that built it; "
            "this is how a reader checks a row against the HLO instead of trusting it."
        ),
        "untraced_wall_ms": untraced_wall_ms,
        "reconciliation_vs_untraced_pct": (
            100.0 * (sum_ms - untraced_wall_ms) / untraced_wall_ms if untraced_wall_ms else None
        ),
        "wall_basis": (
            "wall_ms is the TRACED run's wall, the run these kernel durations came from. "
            "untraced_wall_ms is the same executable without the profiler; the difference is "
            "the profiler's overhead and belongs in neither the stages nor the idle row."
        ),
        "calls": int(calls),
        "call_split": split_method,
        "call_blocks": n_blocks,
        "events": len(events),
        "streams": sorted({e["stream"] for e in events}),
        "per_stage_ms": per_stage_ms,
        "stage_order": [label for label in STAGE_ORDER if label in per_stage_ms]
        + [label for label in sorted(per_stage_ms) if label not in STAGE_ORDER],
        "mixed_fusion": {
            "ms": per_stage_ms.get(MIXED_FUSION, {}).get("median_ms", 0.0),
            "constituent_stage_sets": {
                " + ".join(k): v for k, v in sorted(mixed_sets.items(), key=lambda kv: -kv[1])
            },
            "example_kernels": {
                " + ".join(k): mixed_examples.get(k, [])[:4]
                for k, _ in sorted(mixed_sets.items(), key=lambda kv: -kv[1])[:10]
            },
            "prorated_estimate_ms": dict(sorted(prorated.items(), key=lambda kv: -kv[1])),
            "prorated_note": (
                "ESTIMATE ONLY — a mixed fusion's time split equally across the stages its "
                "constituents span. Never summed into the measured table, never reconciled."
            ),
        },
        "other_sources_ms": dict(sorted(other_sources.items(), key=lambda kv: -kv[1])),
        "device_span_ms": device_span_ms,
        "device_busy_ms": device_busy_ms,
        "device_idle_ms": device_idle_ms,
        "largest_idle_gaps": gaps[:8],
        "kernel_ms": kernel_ms,
        "sum_ms": sum_ms,
        "wall_ms": wall_ms,
        "host_outside_span_ms": host_outside_span_ms,
        "reconciliation_pct": reconciliation_pct,
        "unjoined_ms": unjoined_ms,
        "unjoined_pct": 100.0 * unjoined_ms / wall_ms if wall_ms else float("nan"),
        "other_ms": other_ms,
        "other_pct": 100.0 * other_ms / wall_ms if wall_ms else float("nan"),
        "note": (
            "Stage rows attribute the command-buffers-OFF program (see the module docstring): "
            "kernel times are the same, the launch overhead the CUDA graphs remove is not. "
            "sum_ms = kernel rows + device_idle_ms; the residual against wall_ms is "
            "host_outside_span_ms (dispatch before the first kernel, teardown after the last)."
        ),
    }


# ===========================================================================
# PART 6 — the HLO census
# ===========================================================================


#: Opcodes that move or relabel a value without computing anything. Walking back
#: from a consumer means walking THROUGH these, or every answer is "a bitcast".
PASSTHROUGH_OPCODES = frozenset({"bitcast", "copy", "get-tuple-element", "reshape"})


def callers_of_computations(index: Mapping[str, Instruction]) -> dict[str, list[Instruction]]:
    """``{computation_name: [instructions that ``calls=`` it]}``."""
    callers: dict[str, list[Instruction]] = {}
    for instruction in index.values():
        for name in instruction.calls:
            callers.setdefault(name, []).append(instruction)
    return callers


def resolve_producer(
    index: Mapping[str, Instruction],
    name: str,
    callers: Mapping[str, list[Instruction]] | None = None,
    max_hops: int = 32,
) -> Instruction | None:
    """Walk back from *name* to the instruction that actually computes the value.

    Two kinds of hop are taken. A :data:`PASSTHROUGH_OPCODES` instruction is
    replaced by its operand — otherwise every producer in a GPU module answers
    "a bitcast". A ``parameter`` of a fused computation is replaced by the
    corresponding operand of the fusion that calls it, which is the hop that
    leaves the fusion and finds the value's real origin; it is only taken when
    exactly one instruction calls that computation, because with several the
    parameter has several origins and naming one would be a guess.

    Returns the first instruction that is neither, or the last one reached when
    the chain cannot be followed further.
    """
    if callers is None:
        callers = callers_of_computations(index)
    current = index.get(name)
    seen: set[str] = set()
    while current is not None and current.name not in seen and max_hops > 0:
        seen.add(current.name)
        max_hops -= 1
        if current.opcode in PASSTHROUGH_OPCODES and current.operands:
            following = index.get(current.operands[0])
        elif current.opcode == "parameter" and current.parameter_index is not None:
            calling = callers.get(current.computation, [])
            following = None
            if len(calling) == 1 and current.parameter_index < len(calling[0].operands):
                following = index.get(calling[0].operands[current.parameter_index])
        else:
            return current
        if following is None:
            return current
        current = following
    return current


def _where_the_sum_lives(index: Mapping[str, Instruction], written_at, abstract: str) -> dict:
    """Locate ``F + lambda*H`` by its CONSUMERS, once XLA has fused the sum away.

    No standalone ``add`` survives at ``abstract.py:371`` on the GPU leg, so the
    matrix cannot be found by looking for the sum. It can be found by looking for
    the two things that consume it — the edge subset
    (``abstract.py:606-614``) and the log-determinant Cholesky
    (``_log_det_symmetric_from`` called from ``log_det_curvature_reg_matrix_term``)
    — and walking one operand back. That names the fusion that actually
    materialises ``F + lambda*H``, and whether the two consumers share it.
    """
    consumers: list[Instruction] = []
    for instruction in index.values():
        if written_at(instruction, abstract, (606, 614)) and instruction.opcode in (
            "gather",
            "fusion",
            "slice",
            "dynamic-slice",
        ):
            consumers.append(instruction)
        elif (
            instruction.opcode in ("cholesky", "custom-call")
            and instruction.name.split(".")[0] == "cholesky"
            and any(
                "inversion/inversion/abstract.py" in f.file.replace("\\", "/")
                and 884 <= f.line <= 896
                for f in instruction.frames
            )
        ):
            consumers.append(instruction)

    callers = callers_of_computations(index)
    rows = []
    producers: dict[str, list[str]] = {}
    for consumer in consumers:
        for operand_name in consumer.operands:
            direct = index.get(operand_name)
            # F + lambda*H is square. The other operands of these consumers are
            # index vectors and are not the matrix.
            if direct is None or len(direct.dims) != 2 or direct.dims[0] != direct.dims[1]:
                continue
            producer = resolve_producer(index, operand_name, callers)
            if producer is None:
                continue
            rows.append(
                {
                    "consumer": consumer.name,
                    "consumer_opcode": consumer.opcode,
                    "consumer_source": consumer.source,
                    "operand": operand_name,
                    "producer": producer.name,
                    "producer_opcode": producer.opcode,
                    "producer_shape": producer.shape,
                    "producer_computation": producer.computation,
                    "producer_calls": list(producer.calls),
                    "producer_source": producer.source,
                }
            )
            producers.setdefault(producer.name, []).append(consumer.name)

    shared = sorted(name for name, users in producers.items() if len(set(users)) > 1)
    return {
        "consumers_of_F_plus_lambda_H": rows,
        "producers": {k: sorted(set(v)) for k, v in producers.items()},
        "producers_shared_by_more_than_one_consumer": shared,
        "note": (
            "The F + lambda*H matrix is located by its consumers (the edge subset at "
            "abstract.py:606-614 and the log-det Cholesky reached from "
            "log_det_curvature_reg_matrix_term) and one step back along their operands, "
            "because XLA leaves no standalone add at abstract.py:371 to find. A producer "
            "listed under producers_shared_by_more_than_one_consumer is ONE materialisation "
            "serving both, which is the JAX-side answer to "
            "curvature_reg_matrix_rebuilt_every_access; distinct producers per consumer "
            "would be the sum being built more than once."
        ),
    }


def _curvature_verdict(adds: Sequence[Instruction], opcodes_at_line: Mapping[str, int]) -> str:
    """The ``F + lambda*H`` verdict, from the adds AND the opcodes at the line.

    Three different situations produce "no add", and only one of them is a bug:

    - **Nothing at all resolves to the line.** The line number has moved in
      PyAutoArray and the census is measuring nothing. Say so.
    - **The line resolves but no ``add`` survives.** XLA fused the sum into its
      consumers; no standalone ``F + lambda*H`` matrix is ever materialised. On
      the RTX leg ``abstract.py:371`` carried exactly one instruction — a
      ``transpose`` — and no add. That is the JAX-side answer to the
      ``curvature_reg_matrix_rebuilt_every_access`` draft, and it is not the
      same as "the line number is wrong".
    - **Adds survive.** Then the question is how many distinct computations they
      live in, because XLA tiles: counting instructions cannot tell one tiled sum
      from two sums.
    """
    if not adds and not opcodes_at_line:
        return (
            "NO instruction of any opcode resolves to abstract.py:371 — the line number has "
            "moved in the installed PyAutoArray and this census is measuring nothing. Fix the "
            "line before reading anything into it."
        )
    if not adds:
        seen = ", ".join(f"{op} x{n}" for op, n in sorted(opcodes_at_line.items()))
        return (
            f"NO add survives at abstract.py:371, though the line does resolve ({seen}). XLA "
            f"fused the F + lambda*H sum into its consumers: no standalone (n,n) sum is ever "
            f"materialised on the JAX path, so 'the matrix is rebuilt on every access' has no "
            f"cost to measure here. Its arithmetic is inside the mixed_fusion and "
            f"curvature_reg_add rows of the device table, not in an add of its own."
        )
    computations = sorted({i.computation for i in adds})
    callers = sorted({tuple(str(f) for f in i.frames[1:4]) for i in adds})
    if len(computations) == 1:
        return (
            f"{len(adds)} add instruction(s) written at abstract.py:371, all in ONE computation "
            f"({computations[0]}) reached from {len(callers)} distinct caller chain(s) — XLA "
            f"tiles the sum, so this is ONE F + lambda*H, not one per access"
        )
    return (
        f"{len(adds)} add instruction(s) written at abstract.py:371 across "
        f"{len(computations)} distinct computations ({', '.join(computations[:4])}) and "
        f"{len(callers)} distinct caller chain(s) — inspect 'callers' before concluding the "
        f"matrix is built more than once; separate computations can still be one sum tiled "
        f"across a fusion and its parent"
    )


def hlo_census(index: Mapping[str, Instruction]) -> dict:
    """Count the instructions phase 1 was commissioned to count.

    - **(n,n) ``add`` at ``inversion/inversion/abstract.py:371``** — the
      ``F + lambda*H`` sum. ``curvature_reg_matrix`` is a plain property reached
      twice per evaluation (once at :613 through the edge subset, once at :397
      through ``curvature_reg_matrix_reduced``), so 1 means XLA's CSE collapsed
      the two and 2 means it did not. This is the JAX-side verdict on the
      ``curvature_reg_matrix_rebuilt_every_access`` draft.
    - **gathers at :613 and :397** — the two ``[ids][:, ids]`` subsets.
    - **FFT instructions sourced at ``operators/convolver.py``**, split by whether
      the stack reaches the mapping-matrix convolution or the image one. Two
      mapping-matrix FFT groups would mean ``operated_mapping_matrix_list`` (a
      plain property reached twice) convolves the cube twice.

    Every count carries the instruction names it counted, so the number can be
    checked against the HLO rather than trusted.
    """

    def _hits(predicate) -> list[Instruction]:
        return sorted(
            (i for i in index.values() if predicate(i)), key=lambda i: (i.computation, i.name)
        )

    def _called_from(instruction: Instruction, path: str, lines: tuple[int, int]) -> bool:
        """Does ANY frame of the stack sit at *path* within *lines*? (caller identity)"""
        return any(
            path in f.file.replace("\\", "/") and lines[0] <= f.line <= lines[1]
            for f in instruction.frames
        )

    def _written_at(instruction: Instruction, path: str, lines: tuple[int, int]) -> bool:
        """Is the INNERMOST frame at *path* within *lines*? (where the op was written)

        This distinction is the whole census. ``curvature_reg_matrix`` is
        ``self._xp.add(self.curvature_matrix, self.regularization_matrix)`` on
        ONE line, and Python evaluates both arguments *at that line* — so
        ``abstract.py:371`` appears in the stack of every instruction downstream
        of it: the Delaunay interpolation weights, the split regularization
        matrix, the adaptive pixel signals, all of it. Scanning the whole stack
        for :371 counted 15 "F + lambda*H adds" whose innermost frames were in
        ``delaunay.py``, ``regularization_util.py`` and ``mapper_util.py``.
        Only the innermost frame says where an instruction was actually written.
        """
        if not instruction.frames:
            return False
        frame = instruction.frames[0]
        return path in frame.file.replace("\\", "/") and lines[0] <= frame.line <= lines[1]

    abstract = _ARRAY + _INVERSION_ABSTRACT

    # Every add traced at :371, square or not. XLA fuses and slices, so a
    # constituent of the F + lambda*H sum can carry a tile shape rather than
    # (n,n); filtering on (n,n) first made the count 0 on a shakeout whose mixed
    # fusions plainly contained the add. The square ones are reported separately.
    curvature_reg_adds = _hits(lambda i: i.opcode == "add" and _written_at(i, abstract, (371, 371)))
    # Every opcode written at :371, so "no add survives" can be told apart from
    # "the line number has moved". They are different findings.
    opcodes_at_371: dict[str, int] = {}
    for _instruction in index.values():
        if _written_at(_instruction, abstract, (371, 371)):
            opcodes_at_371[_instruction.opcode] = opcodes_at_371.get(_instruction.opcode, 0) + 1
    instructions_at_371 = _hits(lambda i: _written_at(i, abstract, (371, 371)))
    curvature_reg_adds_square = [
        i for i in curvature_reg_adds if len(i.dims) == 2 and i.dims[0] == i.dims[1]
    ]
    edge_gathers = _hits(
        lambda i: (
            i.opcode in ("gather", "dynamic-slice", "slice", "concatenate")
            and _written_at(i, abstract, (606, 614))
        )
    )
    reduced_gathers = _hits(
        lambda i: (
            i.opcode in ("gather", "dynamic-slice", "slice", "concatenate")
            and _written_at(i, abstract, (374, 398))
        )
    )

    # An FFT or a convolution counts when it was WRITTEN at convolver.py or at the
    # phase-3 harness (``psf_cube_injection``, #295): a candidate's own rfft2 /
    # lax.conv has its innermost frame in the harness, not in the library.
    def _written_at_convolution_source(i: Instruction) -> bool:
        return _written_at(i, _ARRAY + _CONVOLVER, (1, 10**6)) or _written_at(
            i, _PSF_CUBE_HARNESS, (1, 10**6)
        )

    def _is_mapping_convolution(i: Instruction) -> bool:
        return (
            _called_from(i, _ARRAY + _IMAGING_ABSTRACT, (119, 150))
            or _called_from(i, _ARRAY + _CONVOLVER, (670, 691))
            or _called_from(i, _ARRAY + _CONVOLVER, (1086, 1300))
            or _called_from(i, _PSF_CUBE_HARNESS, (1, 10**6))
        )

    fft_all = _hits(lambda i: i.opcode == "fft" and _written_at_convolution_source(i))
    fft_mapping = [i for i in fft_all if _is_mapping_convolution(i)]
    fft_image = [i for i in fft_all if i not in fft_mapping]
    # A real-space candidate (``real_space_direct``, ``conv_cudnn_batched``) has no
    # fft at all: its work is a ``convolution`` opcode, which the GPU backend
    # rewrites into a cuDNN ``custom-call`` named ``cudnn-conv*``.
    conv_mapping = _hits(
        lambda i: (
            (i.opcode == "convolution" or (i.opcode == "custom-call" and "conv" in i.name))
            and _written_at_convolution_source(i)
            and _is_mapping_convolution(i)
        )
    )

    # A factorization is either the ``cholesky`` opcode or, on the GPU backend,
    # a cuSOLVER ``custom-call`` that XLA names ``cholesky.N``. Matching
    # ``"cholesky" in op_name`` instead swept in every instruction traced inside
    # jnp.linalg.cholesky's own jit (103 of them on the shakeout), which counts a
    # helper's internals; matching the opcode alone found none (0), because the
    # real factorizations are custom-calls.
    def _is_factorization(i: Instruction) -> bool:
        return i.opcode == "cholesky" or (
            i.opcode == "custom-call" and i.name.split(".")[0] == "cholesky"
        )

    cholesky = _hits(_is_factorization)
    triangular_solves = _hits(
        lambda i: (
            i.opcode == "triangular-solve"
            or (i.opcode == "custom-call" and i.name.split(".")[0] == "triangular-solve")
        )
    )

    def _describe(instructions: Sequence[Instruction]) -> dict:
        return {
            "count": len(instructions),
            "instructions": [
                {
                    "name": i.name,
                    "opcode": i.opcode,
                    "shape": i.shape,
                    "computation": i.computation,
                    "source": i.source,
                    "stack": [str(f) for f in i.frames[:4]],
                }
                for i in instructions
            ],
        }

    return {
        "curvature_reg_add_nn": {
            **_describe(curvature_reg_adds),
            "count_square_nn": len(curvature_reg_adds_square),
            "square_instructions": [i.name for i in curvature_reg_adds_square],
            "computations": sorted({i.computation for i in curvature_reg_adds}),
            "callers": {i.name: [str(f) for f in i.frames[:6]] for i in curvature_reg_adds},
            "question": (
                "How many (n,n) adds of F + lambda*H survive XLA? curvature_reg_matrix is a plain "
                "property reached twice per evaluation (abstract.py:613 and :397). 1 = CSE "
                "collapsed them; 2 = the matrix is built twice on the JAX path."
            ),
            "opcodes_at_line_371": dict(sorted(opcodes_at_371.items())),
            "instructions_at_line_371": [
                {
                    "name": i.name,
                    "opcode": i.opcode,
                    "shape": i.shape,
                    "computation": i.computation,
                }
                for i in instructions_at_371
            ],
            "where_the_sum_lives": _where_the_sum_lives(index, _written_at, abstract),
            "verdict": _curvature_verdict(curvature_reg_adds, opcodes_at_371),
        },
        "edge_subset_gathers_613": _describe(edge_gathers),
        "curvature_reg_reduced_gathers_397": _describe(reduced_gathers),
        "fft_convolver_total": {
            **_describe(fft_all),
            "callers": {i.name: [str(f) for f in i.frames[:5]] for i in fft_all},
            "question": (
                "Each FFT convolution is one rfft2 + one irfft2, so a convolution costs TWO fft "
                "instructions. The caller chain of each is recorded because the same "
                "convolved_mapping_matrix_from is reached from more than one place "
                "(operated_mapping_matrix_list for F, and "
                "mapped_reconstructed_operated_data_dict for the mapped reconstruction) -- "
                "counting instructions without their callers cannot tell a doubled convolution "
                "from two different convolutions."
            ),
        },
        "fft_mapping_matrix": {
            **_describe(fft_mapping),
            "question": (
                "Is the PSF convolution of the mapping-matrix cube compiled once or twice? "
                "operated_mapping_matrix_list is a plain property reached twice."
            ),
        },
        "fft_image": _describe(fft_image),
        "conv_mapping_matrix": {
            **_describe(conv_mapping),
            "question": (
                "How many real-space convolutions of the mapping-matrix cube survive XLA? "
                "0 on the shipped FFT path; a phase-3 real-space candidate shows here instead "
                "of in fft_mapping_matrix (a cuDNN conv is a custom-call on the GPU backend)."
            ),
        },
        "cholesky": {
            **_describe(cholesky),
            "callers": {i.name: [str(f) for f in i.frames[:5]] for i in cholesky},
        },
        "triangular_solve": _describe(triangular_solves),
        "instructions_total": len(index),
        "instructions_with_frames": sum(1 for i in index.values() if i.frames),
    }


@dataclass(frozen=True)
class CensusAnchor:
    """A PyAutoArray source-line anchor the census reads, and the rows it backs.

    ``mode="encloses"``: the census range must ENCLOSE the whole ``function``
    (def line to last line), the shape of the convolver / imaging-abstract ranges.
    ``mode="inside"``: the census range must lie INSIDE ``function``, and when
    ``contains`` is given some line of the range must contain that text — the
    shape of the single-line ``F + lambda*H`` anchor.
    """

    rows: tuple[str, ...]
    path: str
    function: str
    lines: tuple[int, int]
    mode: str = "encloses"
    contains: str | None = None


_PSF_CENSUS_ROWS = (
    "fft_convolver_total",
    "fft_mapping_matrix",
    "fft_image",
    "conv_mapping_matrix",
)

#: Every PyAutoArray line anchor ``hlo_census`` reads. The PSF anchors back the
#: rows phase 3 reads; the ``abstract.py`` anchors back the phase-1 curvature and
#: gather rows (which is why phase 2 excluded the whole census when they moved).
CENSUS_ANCHORS: tuple[CensusAnchor, ...] = (
    CensusAnchor(_PSF_CENSUS_ROWS, _CONVOLVER, "convolved_mapping_matrix_from", (1086, 1300)),
    CensusAnchor(
        _PSF_CENSUS_ROWS, _CONVOLVER, "_convolved_mapping_matrix_over_sampled_jax_from", (670, 691)
    ),
    CensusAnchor(_PSF_CENSUS_ROWS, _IMAGING_ABSTRACT, "operated_mapping_matrix_list", (119, 150)),
    CensusAnchor(
        ("curvature_reg_add_nn",),
        _INVERSION_ABSTRACT,
        "curvature_reg_matrix",
        (371, 371),
        mode="inside",
        contains="add(",
    ),
    CensusAnchor(
        ("curvature_reg_reduced_gathers_397",),
        _INVERSION_ABSTRACT,
        "curvature_reg_matrix_reduced",
        (374, 398),
        mode="inside",
    ),
    CensusAnchor(
        ("edge_subset_gathers_613",),
        _INVERSION_ABSTRACT,
        "reconstruction",
        (606, 614),
        mode="inside",
        contains="ids_to_keep]",
    ),
)


def _function_spans(source: str) -> dict[str, list[tuple[int, int]]]:
    """``{function name: [(def line, last line), ...]}`` for every def in *source*."""
    import ast

    spans: dict[str, list[tuple[int, int]]] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.setdefault(node.name, []).append((node.lineno, node.end_lineno or node.lineno))
    return spans


def census_anchor_status(
    package_root: str | os.PathLike | None = None,
    anchors: Sequence[CensusAnchor] = CENSUS_ANCHORS,
) -> dict:
    """Check the census's line anchors against the INSTALLED PyAutoArray source.

    Phase 2 had to mark the whole census ``excluded`` because the anchors had
    moved and nothing noticed. This makes the check a measurement: each anchor
    is resolved against the source file the running process imports (or the
    ``autoarray`` package directory *package_root*), and the census status is
    ``ok`` only when every anchor backing the PSF rows holds. Rows whose anchors
    have moved are listed under ``rows_excluded`` with the reason — never
    silently reported.
    """
    if package_root is None:
        import importlib.util

        spec = importlib.util.find_spec("autoarray")
        if spec is None or spec.origin is None:
            return {
                "status": "excluded",
                "reason": "autoarray is not importable; no anchor can be verified",
                "anchors": [],
                "rows_excluded": sorted({r for a in anchors for r in a.rows}),
            }
        package_root = os.path.dirname(spec.origin)

    results = []
    sources: dict[str, tuple[list[str], dict]] = {}
    for anchor in anchors:
        path = os.path.join(str(package_root), anchor.path)
        if path not in sources:
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
                sources[path] = (text.splitlines(), _function_spans(text))
            except OSError as error:
                sources[path] = ([], {"__error__": str(error)})
        lines, spans = sources[path]
        found = spans.get(anchor.function, [])
        first, last = anchor.lines
        ok = False
        reason = ""
        if not found:
            reason = f"def {anchor.function} not found in {anchor.path}"
        elif anchor.mode == "encloses":
            ok = any(first <= a <= b <= last for a, b in found)
            if not ok:
                reason = (
                    f"def {anchor.function} spans {found}, not enclosed by the census range "
                    f"{first}-{last}"
                )
        else:
            ok = any(a <= first and last <= b for a, b in found)
            if not ok:
                reason = (
                    f"the census range {first}-{last} is not inside def {anchor.function} "
                    f"(spans {found})"
                )
            elif anchor.contains is not None:
                window = lines[first - 1 : last]
                ok = any(anchor.contains in line for line in window)
                if not ok:
                    reason = (
                        f"no line in {first}-{last} of {anchor.path} contains "
                        f"{anchor.contains!r}; the anchored instruction has moved"
                    )
        results.append(
            {
                "rows": list(anchor.rows),
                "path": anchor.path,
                "function": anchor.function,
                "lines": [first, last],
                "mode": anchor.mode,
                "found_spans": [list(span) for span in found],
                "ok": ok,
                "reason": reason or "ok",
            }
        )

    psf_ok = all(r["ok"] for r in results if set(r["rows"]) & set(_PSF_CENSUS_ROWS))
    rows_excluded = sorted({row for r in results if not r["ok"] for row in r["rows"]})
    return {
        "status": "ok" if psf_ok else "excluded",
        "status_scope": (
            "the PSF rows (fft_convolver_total, fft_mapping_matrix, fft_image, "
            "conv_mapping_matrix); any other row whose anchor moved is in rows_excluded"
        ),
        "reason": (
            "every PSF census anchor verified against the installed PyAutoArray"
            if psf_ok
            else "a PSF census anchor moved in the installed PyAutoArray: "
            + "; ".join(r["reason"] for r in results if not r["ok"])
        ),
        "package_root": str(package_root),
        "anchors": results,
        "rows_excluded": rows_excluded,
    }


def stage_map_provenance(stage_map: Sequence[StageRule] = STAGE_MAP) -> list[dict]:
    """The stage map as data, written into every leg's JSON.

    Line numbers are read from a specific PyAutoArray / PyAutoLens revision. A
    later library edit that moves them does not silently produce a wrong table —
    it produces a growing ``other`` row — but only if the map that produced a
    number is recoverable from the number's own JSON.
    """
    return [
        {
            "label": rule.label,
            "path": rule.path,
            "lines": list(rule.lines) if rule.lines else None,
            "requires": (
                {
                    "path": rule.requires[0],
                    "lines": list(rule.requires[1]) if rule.requires[1] else None,
                }
                if rule.requires
                else None
            ),
            "function": rule.function,
        }
        for rule in stage_map
    ]

"""Access-counting instrumentation for a single library likelihood call.

Why this module exists
----------------------

The numba CPU cells decompose a likelihood by *timing accessor by accessor*:
``delaunay_numba.py`` builds a fresh ``FitImaging`` per repeat and touches each
lazy cached property in dependency order, so each timing isolates one step's
incremental cost. That works, but it measures a **reconstruction** of the call,
not the call: the cell decides the order, the cell decides which properties
exist, and a step the library reaches twice (or not at all) is invisible. Two of
that cell's rows are already reported as *residuals* for exactly this reason.

This module measures the call itself. :func:`install` replaces named descriptors
(and module-level functions) with timing wrappers, the production
``log_likelihood_function`` is then run untouched, and :func:`snapshot` reports,
per site:

- ``incl_s``  — inclusive time: everything that happened inside the site;
- ``excl_s``  — exclusive (self) time: inclusive minus the time spent inside
  *nested wrapped* sites;
- ``n_calls`` — how many times the library actually reached the site.

Exclusive times are additive **by construction**: every wrapped entry pushes a
frame onto a stack and hands its whole inclusive elapsed time up to its parent's
child-accumulator, so ``sum(excl_s)`` is the wrapped fraction of the top-level
call and ``call - sum(excl_s)`` is the honest unattributed remainder. Nothing is
estimated and nothing is a residual.

Recursion (the same site re-entered beneath itself) adds its elapsed time to
``excl_s`` at every depth but to ``incl_s`` only at the outermost frame, so a
re-entrant site cannot double-count its own inclusive time while the exclusive
sum stays additive.

Three descriptor flavours, which are not the same object
--------------------------------------------------------

- :class:`functools.cached_property` and ``autonerves.CachedProperty``
  (``PyAutoNerves/autonerves/tools/decorators.py:6-23``) are **non-data**
  descriptors: after the first access the computed value lives in
  ``obj.__dict__`` under the attribute name and ``__get__`` is never called
  again. A site declared ``cached=True`` must therefore report ``n_calls == 1``
  for one likelihood call — if it reports more, the library stopped caching it
  and the decomposition means something different. The two classes are
  unrelated (one is stdlib, one is a hand-rolled bottle.py descriptor), so both
  are handled by delegation rather than by isinstance.
- A plain ``property`` is a **data** descriptor: it wins over ``obj.__dict__``
  and is therefore seen on *every* access. ``AbstractInversion.curvature_reg_matrix``
  is one of these, and the fact that the library recomputes it several times per
  call is a measurement, not a bug to paper over.

The wrapper preserves the original's data-ness: a data descriptor is wrapped in
a wrapper that also defines ``__set__``/``__delete__``, a non-data descriptor in
one that does not, so instance-dict shadowing keeps working exactly as before.

Module-level functions
----------------------

``install`` also wraps plain module attributes, for
``autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from``
(``inversion_util.py:256``) and ``autoarray.util.fnnls.fnnls_cholesky``
(``autoarray/util/fnnls.py:27``). ``abstract.py`` imports the *module* and
resolves the attribute at call time (the pattern
``library_solver_injection.py:23-30`` documents), so rebinding the module
attribute is picked up by the next call and restored on ``uninstall``.

Those two also carry the solve's diagnostics: ``fnnls_cholesky`` fills a
``stats`` dict passed in by its caller, and ``reconstruction_positive_only_from``
adds ``seed_source`` / ``warm_start_fallback`` to that same dict *after* the
solver returns (``inversion_util.py:305-312``). The wrapper keeps a reference to
the dict it saw rather than a copy, so :func:`captured_stats` reads the finished
article and the cell reports ``outer_iterations`` / ``inner_iterations`` /
``passive_set`` without a second, untimed solve.

No JAX
------

stdlib + numpy only, at every level. These rows exist to measure what a CPU
numba likelihood costs, and the cell that uses this module asserts
``"jax" not in sys.modules`` after its imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Site",
    "captured_stats",
    "declared_cached_labels",
    "descriptor_site",
    "function_site",
    "install",
    "installed_labels",
    "reset",
    "snapshot",
    "uninstall",
]


#: The ``stats`` keys the two wrapped solver functions populate, in the order
#: the cell reports them. ``passive_set`` is an index array and is summarised by
#: its length plus the array itself, so P3 can count the passive-set differences
#: between the dense and the sparse-numba legs.
STATS_KEYS = (
    "seed_source",
    "warm_start_fallback",
    "outer_iterations",
    "inner_iterations",
    "passive_set",
    "n_passive",
    "warm_start_errors",
)


@dataclass(frozen=True)
class Site:
    """One instrumented access point.

    ``target`` is a class (for ``kind="descriptor"``) or a module (for
    ``kind="function"``). ``cached`` declares that the library caches this site
    per object, which the cell turns into a hard ``n_calls == 1`` gate.
    ``missing_ok`` marks a site that only exists on one formalism: it is
    registered with zeroed counters rather than raising, so a row can report
    ``n_calls: 0`` instead of the label silently vanishing.
    """

    label: str
    target: Any
    name: str
    kind: str = "descriptor"
    cached: bool = False
    missing_ok: bool = False
    group: str | None = None

    def __post_init__(self):
        if self.kind not in ("descriptor", "function"):
            raise ValueError(f"{self.label}: kind must be 'descriptor' or 'function'")


def descriptor_site(label, cls, name, *, cached=False, missing_ok=False, group=None) -> Site:
    """A ``property`` / ``cached_property`` access point on ``cls``."""
    return Site(
        label=label,
        target=cls,
        name=name,
        kind="descriptor",
        cached=cached,
        missing_ok=missing_ok,
        group=group,
    )


def function_site(label, module, name, *, missing_ok=False, group=None) -> Site:
    """A module-level function, rebound on ``module`` for the duration."""
    return Site(
        label=label,
        target=module,
        name=name,
        kind="function",
        cached=False,
        missing_ok=missing_ok,
        group=group,
    )


@dataclass
class _Record:
    incl_s: float = 0.0
    excl_s: float = 0.0
    n_calls: int = 0
    depth: int = 0
    stats: dict | None = field(default=None, repr=False)


#: label -> _Record. Registered by ``install`` (including for missing sites, so
#: their zeros are reported rather than absent) and zeroed by ``reset``.
_records: dict[str, _Record] = {}

#: The child-time accumulator stack. One float per live wrapped frame, holding
#: the summed *inclusive* time of the wrapped entries that closed inside it.
_stack: list[float] = []

#: ``(owner, name, original, site)`` per installed site, in install order.
_installed: list[tuple[Any, str, Any, Site]] = []

#: label -> Site, for every registered site including the missing ones.
_sites: dict[str, Site] = {}


def _timed(label: str, call, *args, **kwargs):
    """Run ``call``, charging its time to ``label`` inclusively and exclusively."""
    record = _records[label]
    record.n_calls += 1
    record.depth += 1
    _stack.append(0.0)
    start = time.perf_counter()
    try:
        return call(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        children = _stack.pop()
        record.depth -= 1
        # Inclusive time only at the outermost frame of a re-entrant site, so a
        # recursive site cannot count its own inclusive time twice. Exclusive
        # time accrues at every depth, which is what keeps the sum additive.
        if record.depth == 0:
            record.incl_s += elapsed
        record.excl_s += elapsed - children
        if _stack:
            _stack[-1] += elapsed


class _TimedNonDataDescriptor:
    """Wrapper for a non-data descriptor (both ``cached_property`` flavours).

    Defines only ``__get__``, so an instance-dict entry written by the wrapped
    descriptor keeps shadowing it: the site is seen exactly once per object,
    which is what makes the ``n_calls == 1`` gate meaningful.
    """

    def __init__(self, label: str, original):
        self.label = label
        self.original = original
        self.__doc__ = getattr(original, "__doc__", None)

    def __set_name__(self, owner, name):
        setter = getattr(self.original, "__set_name__", None)
        if setter is not None:
            setter(owner, name)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.original.__get__(obj, objtype)
        return _timed(self.label, self.original.__get__, obj, objtype)


class _TimedDataDescriptor(_TimedNonDataDescriptor):
    """Wrapper for a data descriptor (a plain ``property``).

    ``__set__`` / ``__delete__`` are delegated untimed — they are not accesses
    the likelihood makes, and defining them is what preserves the original's
    precedence over ``obj.__dict__``.
    """

    def __set__(self, obj, value):
        self.original.__set__(obj, value)

    def __delete__(self, obj):
        self.original.__delete__(obj)


def _timed_function(label: str, original):
    """A module-level function wrapper that also keeps the ``stats`` dict it saw."""

    def wrapper(*args, **kwargs):
        stats = kwargs.get("stats")
        if stats is not None:
            # The reference, not a copy: `reconstruction_positive_only_from`
            # writes `seed_source` / `warm_start_fallback` into this same dict
            # after the solver returns (inversion_util.py:305-312).
            _records[label].stats = stats
        return _timed(label, original, *args, **kwargs)

    wrapper.__name__ = getattr(original, "__name__", label)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper.__wrapped__ = original
    return wrapper


def _descriptor_owner(cls, name):
    """The class in ``cls``'s MRO that actually declares ``name``, or ``None``.

    Installing on ``cls`` when the descriptor is declared on a base would
    *shadow* the base's entry, and ``uninstall`` would then leave a permanent
    subclass attribute behind. The wrapper therefore always replaces the real
    declaration and puts the identical object back.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


def install(spec) -> None:
    """Replace every site in ``spec`` with a timing wrapper.

    ``spec`` is any iterable of :class:`Site`. Raises if an installation is
    already live — a double install would wrap the wrappers and report each
    site's time twice.
    """
    if _installed:
        raise RuntimeError(
            "call_accounting.install: an installation is already live; call "
            "uninstall() before installing a second spec."
        )

    spec = list(spec)
    labels = [site.label for site in spec]
    duplicates = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        raise ValueError(f"call_accounting.install: duplicate site labels {sorted(duplicates)}")

    # A previous spec's labels must not survive into this snapshot: the row set
    # of a decomposition is exactly the spec that was installed for it.
    _records.clear()
    _sites.clear()
    _stack.clear()

    try:
        for site in spec:
            _records[site.label] = _Record()
            _sites[site.label] = site

            if site.kind == "descriptor":
                owner = _descriptor_owner(site.target, site.name)
                if owner is None:
                    if site.missing_ok:
                        continue
                    raise AttributeError(
                        f"call_accounting: {site.target.__name__} has no descriptor "
                        f"{site.name!r} anywhere in its MRO (site {site.label!r})."
                    )
                original = owner.__dict__[site.name]
                is_data = hasattr(original, "__set__") or hasattr(original, "__delete__")
                cls = _TimedDataDescriptor if is_data else _TimedNonDataDescriptor
                setattr(owner, site.name, cls(site.label, original))
                _installed.append((owner, site.name, original, site))
            else:
                original = getattr(site.target, site.name, None)
                if original is None:
                    if site.missing_ok:
                        continue
                    raise AttributeError(
                        f"call_accounting: module {site.target.__name__} has no "
                        f"function {site.name!r} (site {site.label!r})."
                    )
                setattr(site.target, site.name, _timed_function(site.label, original))
                _installed.append((site.target, site.name, original, site))
    except Exception:
        uninstall()
        raise


def uninstall() -> None:
    """Restore every original **by identity**, in reverse install order."""
    while _installed:
        owner, name, original, _site = _installed.pop()
        setattr(owner, name, original)
    _stack.clear()


def reset() -> None:
    """Zero every counter, keeping the installation live."""
    for label in _records:
        _records[label] = _Record()
    _stack.clear()


def snapshot() -> dict[str, dict]:
    """``{label: {excl_s, incl_s, n_calls}}`` for every registered site.

    A site that does not exist on this formalism (``missing_ok``) is present
    with zeros rather than absent, so a decomposition row set is the same set of
    labels on both legs and ``n_calls: 0`` is a readable fact.
    """
    return {
        label: {
            "excl_s": record.excl_s,
            "incl_s": record.incl_s,
            "n_calls": record.n_calls,
        }
        for label, record in _records.items()
    }


def captured_stats() -> dict[str, dict]:
    """The ``stats`` dicts the wrapped solver functions were handed, by label.

    Read after the call, so the keys the caller adds *after* the solver returns
    (``seed_source``, ``warm_start_fallback``) are present. ``passive_set`` is
    returned as it stands; the cell is responsible for any array handling.
    """
    out: dict[str, dict] = {}
    for label, record in _records.items():
        if record.stats is None:
            continue
        out[label] = {key: record.stats[key] for key in STATS_KEYS if key in record.stats}
    return out


def installed_labels() -> list[str]:
    """Labels whose site was actually found and wrapped (missing ones excluded)."""
    return [site.label for _owner, _name, _original, site in _installed]


def declared_cached_labels() -> list[str]:
    """Labels of every registered site declared ``cached=True``."""
    return [label for label, site in _sites.items() if site.cached]


def site_group(label: str) -> str | None:
    """The decomposition group a registered label belongs to, if it declared one."""
    site = _sites.get(label)
    return None if site is None else site.group

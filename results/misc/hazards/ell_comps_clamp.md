# ell_comps clamp — generated seed finding

Semantic finding ID: `component.ell_comps.magnitude-saturation`

`autogalaxy.convert` clamps the ellipticity magnitude at `0.999`, so the
axis ratio saturates at `q = 0.000500250125062532`. The value stays finite; the
hazard is a flat, zero-gradient plateau rather than a NaN rejection.

## Reachability

- NumPy construction reaches the narrow `0.999 <= |ell_comps| < 1` annulus.
- `validate_ell_comps` blocks the scientifically invalid `|ell_comps| >= 1`
  region for concrete NumPy/Python scalars.
- JAX scalar arrays and tracers are not concrete to that guard, so construction
  and tracing both reach the beyond-unit plateau.

That region split matters: `code_exists`, `reachable_via`, `blocked_by`, and
`affects_science` are separate fields in the machine-readable record.

## Prior mass beyond the unit circle

Deterministic Monte Carlo seed `107`;
Wilson 95% intervals are reported rather than a bare percentage.

| Independent prior | Estimate | 95% interval | Samples |
|---|---:|---:|---:|
| TruncatedGaussian(0, 0.3) per component | 0.22% | [0.21%, 0.23%] | 1,000,000 |
| TruncatedGaussian(0, 0.5) per component | 5.1% | [5.0%, 5.1%] | 1,000,000 |
| Uniform(-1, 1) per component | 21.4% | [21.4%, 21.5%] | 1,000,000 |

These reproduce the established rounded results: 0.22%, 5.1%, and 21.4%.

## Code anchors

- `PyAutoGalaxy/autogalaxy/convert.py:71` at commit `be61b8d0546c`
  (`autogalaxy.convert.axis_ratio_and_angle_from`; token fingerprint `244242025d2db16d…`).
- `PyAutoGalaxy/autogalaxy/profiles/validate.py:154` at commit `be61b8d0546c`
  (`autogalaxy.profiles.validate.validate_ell_comps`; token fingerprint `9971e6026aba80d2…`).

The semantic finding ID is stable. Anchors help locate the implementation, but
persistence is decided by re-running the reproducer (`scan.py --check`), not by
treating a source hash as the finding's identity.
